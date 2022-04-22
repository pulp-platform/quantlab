import os
import torch
import torch.nn as nn
import torchvision

from systems.ILSVRC12.MobileNetV1 import MobileNetV1
from systems.ILSVRC12.MobileNetV1.preprocess import TransformA
import quantlib.newediting.graphs as qlg
import quantlib.newediting.editing as qle


def load_ilsvrc12(partition: str, path_data: str, transform: torchvision.transforms.Compose) -> torch.utils.data.Dataset:
    # basic dataset creation function for ILSVRC2012
    path_dataset = os.path.join(os.path.realpath(path_data), 'train' if partition == 'train' else 'val')
    dataset = torchvision.datasets.ImageFolder(path_dataset, transform)
    return dataset


ds = load_ilsvrc12('valid', 'systems/ILSVRC12/data', TransformA(augment=False))
start = 1000
end = 1500

net = MobileNetV1(0.75)
net.load_state_dict(torch.load('systems/ILSVRC12/MobileNetV1/logs/MNv1_0.75_224_relu.ckpt', map_location=torch.device('cpu')))
net.eval()

# TODO: check that
#       1. network lives on CPU
#       2. network is NOT in `training` mode (is this really necessary during F2F conversion?)

gm = qlg.quantlib_fine_symbolic_trace(root=net)
gm.eval()

correct = 0
for i in range(start, end):
    x, y = ds.__getitem__(i)
    x = x.unsqueeze(0)
    yhat = gm(x)
    correct += torch.argmax(yhat, dim=1) == y
print('FP accuracy: ', float(correct) / (end - start))


## FLOAT-2-FAKE CONVERSION

# fake-to-true
f2fspec = [({'types': ('nn.Conv2d', 'nn.Linear')}, {'bitwidth': 8, 'signed': True},  'per-outchannel_weights', 'minmax', 'PACT'),
           ({'types': 'nn.ReLU'},                  {'bitwidth': 8, 'signed': False}, 'per-array',              ('const', {'a': 0.0, 'b': 6.0}),  'PACT')]

actmodulariser  = qle.f2f.ActivationModulariser()
linbnbiasfolder = qle.f2f.LinearBNBiasFolder()
f2fnodewise     = qle.f2f.F2FConverter(f2fspec)
f2fconverter = qle.ComposedEditor([actmodulariser, linbnbiasfolder, f2fnodewise])

gm = f2fconverter(gm)
gm.eval()

correct = 0
for i in range(start, end):
    x, y = ds.__getitem__(i)
    x = x.unsqueeze(0)
    yhat = gm(x)
    correct += torch.argmax(yhat, dim=1) == y
print('F2F (non-init) accuracy: ', float(correct) / (end - start), ' (should be FP accuracy)')


# initialise fake-quantisation...
from quantlib.newalgorithms.qmodules.qmodules.qmodules import _QModule
for n, m in gm.named_modules():
    if isinstance(m, _QModule):
        m.start_observing()
        # m.stop_observing()

for i in range(0, 10):
    x, y = ds.__getitem__(i)
    x = x.unsqueeze(0)
    yhat = gm(x)

from quantlib.newalgorithms.qmodules.qmodules.qmodules import _QModule
for n, m in gm.named_modules():
    if isinstance(m, _QModule):
        # m.start_observing()
        m.stop_observing()

# ...and let's pretend that we trained the model to reasonable accuracy
ckpt = torch.load('systems/ILSVRC12/MobileNetV1/logs/MNv1_0.75_224_INT8w_UINT8x_FQ.ckpt')
gm.load_state_dict(ckpt)

correct = 0
for i in range(start, end):
    x, y = ds.__getitem__(i)
    x = x.unsqueeze(0)
    yhat = gm(x)
    correct += torch.argmax(yhat, dim=1) == y
print('F2F (init) accuracy: ', float(correct) / (end - start))










#################################################
class MNv1Classifier(nn.Module):

    def __init__(self):
        super(MNv1Classifier, self).__init__()
        self.tunnel = qlg.EpsTunnel(torch.Tensor([1.0]))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(1, 1, bias=True)

    def forward(self, x):
        x = self.tunnel(x)
        x = self.avgpool(x)
        x = x.view(-1, 1)
        x = self.linear(x)
        return x


import torch.fx as fx
from typing import List
from quantlib.newutils import quantlib_err_header

class MNv1Finisher(qle.Rewriter):

    def __init__(self):
        name = 'MNv1Finisher'
        super(MNv1Finisher, self).__init__(name)

        self._matcher = qle.LinearGraphMatcher(symbolic_trace_fn=qlg.quantlib_fine_symbolic_trace, pattern_module=MNv1Classifier())

        self._patternname_2_patternnode = {n.target: n for n in filter(lambda n: (n.op in qlg.FXOPCODE_CALL_MODULE), self._matcher.pattern_gm.graph.nodes)}
        self._in_tunnel_node  = self._patternname_2_patternnode['tunnel']
        self._avgpool_node    = self._patternname_2_patternnode['avgpool']
        self._linear_node     = self._patternname_2_patternnode['linear']

    def find(self, g: fx.GraphModule) -> List[qle.ApplicationPoint]:

        candidate_matches = self._matcher.find(g)
        aps = [qle.ApplicationPoint(rewriter=self, graph=g, apcore=match.nodes_map) for match in candidate_matches]
        return aps

    def _check_aps(self, g: fx.GraphModule, aps: List[qle.ApplicationPoint]) -> None:

        # Were the application points computed by this `Rewriter`, and on the target `fx.GraphModule`?
        if not all(map(lambda ap: (ap.rewriter is self) and (ap.graph is g), aps)):
            raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + "can not be applied to application points computed by other Rewritings.")

    def _apply(self, g: fx.GraphModule, ap: qle.ApplicationPoint) -> fx.GraphModule:

        eps = qlg.nnmodule_from_fxnode(ap.apcore[self._in_tunnel_node], g)._eps_out
        old_module = qlg.nnmodule_from_fxnode(ap.apcore[self._linear_node], g)
        new_module = nn.Linear(in_features=old_module.in_features,
                               out_features=old_module.out_features,
                               bias=old_module.bias is not None)

        # TODO: this is incorrect, since the eps is not folded!
        iweight = torch.round(old_module.qweight.data.clone().detach() / old_module.scale.data.clone().detach())
        new_module.weight.data = iweight  # TODO: should I offload the responsibility of computing the true-quantised parameter array to `_QLinear`? Probably yes.
        # Use round, NOT floor: divisions might yield slightly less than the correct unit you are aiming for!
        if new_module.bias is not None:
            new_module.bias.data = old_module.bias.data.clone().detach() / eps

        self._counter += 1
        new_target = '_'.join([self._name.upper(), new_module.__class__.__name__.upper(), str(self._counter)])

        query_in_tunnel = ap.apcore[self._in_tunnel_node]
        query_avgpool   = ap.apcore[self._avgpool_node]
        query_view      = ap.apcore[next(iter(self._avgpool_node.users.keys()))]
        query_linear    = ap.apcore[self._linear_node]

        g.add_submodule(new_target, new_module)
        with g.graph.inserting_after(query_view):
            new_node = g.graph.call_module(new_target, args=(query_view,))
        query_linear.replace_all_uses_with(new_node)

        g.delete_submodule(query_linear.target)
        g.graph.erase_node(query_linear)

        qlg.nnmodule_from_fxnode(ap.apcore[self._in_tunnel_node],  g).set_eps_out(torch.ones_like(qlg.nnmodule_from_fxnode(query_in_tunnel, g)._eps_out))  # TODO: must have the same shape as the input eps

        return g


#################################################











## FAKE-2-TRUE CONVERSION

# TODO: check that
#       1. network lives on CPU
#       2. network is NOT in `training` mode (this IS necessary in F2T conversion)

# annotation
sp = qle.f2t.ShapePropagator()
ep = qle.f2t.EpsPropagator()

gm = sp.apply(gm, {'x': torch.Size((1, 3, 224, 224))})
gm = ep.apply(gm, {'x': 0.020625000819563866})  # constant taken from the PACT/integerise examples (which themselves read it from the ILSVRC12 statistics)

correct = 0
for i in range(start, end):
    x, y = ds.__getitem__(i)
    x = x.unsqueeze(0)
    yhat = gm(x)
    correct += torch.argmax(yhat, dim=1) == y
print('F2T (annotated) accuracy: ', float(correct) / (end - start), ' (should be equal to F2F/init accuracy)')

# add eps-tunnels
eti = qle.f2t.EpsTunnelInserter()
gm = eti(gm)

correct = 0
for i in range(start, end):
    x, y = ds.__getitem__(i)
    x = x.unsqueeze(0)
    x = (x / 0.020625000819563866).floor()
    yhat = gm(x)
    correct += torch.argmax(yhat, dim=1) == y
print('F2T (eps-inserted) accuracy: ', float(correct) / (end - start), ' (should be equal to F2F/init accuracy)')

# integerisation (linear)
loi = qle.f2t.LinearOpsIntegeriser()
gm = loi(gm)

correct = 0
for i in range(start, end):
    x, y = ds.__getitem__(i)
    x = x.unsqueeze(0)
    x = (x / 0.020625000819563866).floor()
    yhat = gm(x)
    correct += torch.argmax(yhat, dim=1) == y
print('F2T (lin-integerised) accuracy: ', float(correct) / (end - start))

# integerisation (requant)
rq = qle.f2t.Requantiser()
gm = rq(gm)

correct = 0
for i in range(start, end):
    x, y = ds.__getitem__(i)
    x = x.unsqueeze(0)
    x = (x / 0.020625000819563866).floor()
    yhat = gm(x)
    correct += torch.argmax(yhat, dim=1) == y
print('F2T (requant-integerised) accuracy: ', float(correct) / (end - start))


#############################
mnv1f = MNv1Finisher()
gm = mnv1f(gm)
#############################


# remove eps-tunnels
etr = qle.f2t.EpsTunnelRemover(force=True)
gm = etr(gm)

correct = 0
for i in range(start, end):
    x, y = ds.__getitem__(i)
    x = x.unsqueeze(0)
    x = (x / 0.020625000819563866).floor()
    yhat = gm(x)
    correct += torch.argmax(yhat, dim=1) == y
print('F2T (eps-removed) accuracy: ', float(correct) / (end - start), ' (should be equal to F2T/requant-integerised accuracy)')


# F2F
# actmodulariser  = qle.f2f.ActivationModulariser()
# linbnbiasfolder = qle.f2f.LinearBNBiasFolder()
# f2fspec = [({'types': ('nn.Conv2d', 'nn.Linear')}, {'bitwidth': 8, 'signed': True},  'per-outchannel_weights', 'minmax', 'PACT'),
#            ({'types': 'nn.ReLU'},                  {'bitwidth': 8, 'signed': False}, 'per-array',              ('const', {'a': 0.0, 'b': 6.0}),  'PACT')]
# f2fnodewise     = qle.f2f.F2FConverter(f2fspec)
# f2fconverter = qle.ComposedEditor([actmodulariser, linbnbiasfolder, f2fnodewise])

# F2T
# sp  = qle.f2t.ShapePropagator()
# ep  = qle.f2t.EpsPropagator()
# eti = qle.f2t.EpsTunnelInserter()
# loi = qle.f2t.LinearOpsIntegeriser()
# rq  = qle.f2t.Requantiser()
# etr = qle.f2t.EpsTunnelRemover()
# f2tconverter = qle.ComposedEditor([sp, ep, eti, loi, gm, etr])
#
# gm = f2tconverter(gm)
