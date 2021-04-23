import torch

import quantlib.graphs as qg
from backends.twn_accelerator.grrules import FoldSTEINQConvBNSTETypeARule, FoldSTEINQConvBNSTETypeBRule, FoldConvBNSTERule, FoldSTEINQConvBNRule


loader = qg.edit.Loader('ImageNet', 'VGG', {})

onnxg = qg.graphs.ONNXGraph(loader.net, torch.ones(1, 3, 224, 224).to('cpu'))
onnxe = qg.editor.Editor(onnxg)
onnxe.startup()
for mod_name, rho in qg.grrules.load_rescoping_rules(modules=['AdaptiveAvgPool2d', 'ViewFlattenNd', 'INQConv2d', 'STEActivation']).items():
    print("Applying rule {} to `nn.Module`s of type {} ...".format(type(rho), mod_name))
    onnxe.set_grr(rho)
    onnxe.edit()
onnxe.shutdown()

pytorchg = qg.graphs.PyTorchGraph(loader.net, onnxg)
pytorche = qg.editor.Editor(pytorchg, onlykernel=True)
pytorche.startup()
# add helper nodes
pytorche.set_grr(qg.grrules.AddInputNodeRule())
pytorche.edit(gs=pytorche.seek(VIs=[{'O000000'}]))
pytorche.set_grr(qg.grrules.AddOutputNodeRule())
pytorche.edit(gs=pytorche.seek(VIs=[{'O000074'}]))
pytorche.set_grr(qg.grrules.AddPrecisionTunnelRule('STEActivation'))
pytorche.edit()
# core editing
pytorche.set_grr(FoldSTEINQConvBNSTETypeARule(beta_frac_bits=17))
pytorche.edit()
pytorche.set_grr(FoldSTEINQConvBNSTETypeBRule(beta_frac_bits=17))
pytorche.edit()
pytorche.set_grr(FoldConvBNSTERule())
pytorche.edit()
pytorche.set_grr(FoldSTEINQConvBNRule())
pytorche.edit()
# remove helper nodes
pytorche.set_grr(qg.grrules.RemovePrecisionTunnelRule())
pytorche.edit()
pytorche.set_grr(qg.grrules.RemoveOutputNodeRule())
pytorche.edit()
pytorche.set_grr(qg.grrules.RemoveInputNodeRule())
pytorche.edit()
# # pytorche.shutdown()

from networkx.algorithms import connected
assert connected.number_connected_components(pytorche.G.to_undirected()) == 1

from networkx.algorithms import dag
import torch.nn as nn
import torch

dummy_input = torch.randn(1, 3, 224, 224)
xold = dummy_input
xnew = dummy_input

imaxold = 0
imaxnew = 0


def get_network(pytorche, version, first=False):

    net = None

    history    = pytorche._history
    stack      = history._undo
    n_versions = len(stack)

    if not first:
        if -n_versions <= version < n_versions:
            G   = stack[version].Gprime
            nd  = stack[version].nodes_dict
            net = nn.Sequential(*[nd[n].nobj for n in dag.topological_sort(G)]).float()
        else:
            print("The required version does not exist!")
    else:
        G   = history._nx_graph
        nd  = history._nodes_dict
        net = nn.Sequential(*[nd[n].nobj for n in dag.topological_sort(G)])

    return net


def propagate_input_to(net, imax, x):
    for i, (n, m) in enumerate(net.named_children()):
        if i <= imax:
            x = m(x)
    return x


def get_ste_layer_eps(net, i):
    ni = net[i].num_levels
    mi = net[i].abs_max_value
    return (2 * mi) / (ni - 1)


def show_net(net):
    for n, m in net.named_children():
        print(n, m)


oldnet = get_network(pytorche, 0, first=True)
newnet = get_network(pytorche, -1)

# for i in range(10):
#     dummy_input = torch.randn(1, 3, 224, 224)
#     ynew = torch.argmax(newnet(dummy_input)).item()
#     yold = torch.argmax(oldnet(dummy_input)).item()
#     print("Trial {} - Old: {} vs. New. {}".format(i, yold, ynew))
