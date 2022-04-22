from systems.ILSVRC12.MobileNetV2 import MobileNetV2
from systems.ILSVRC12.ResNet import ResNet


#############
# USE CASES #
#############

# two use cases:
#   * ResNet18
rn18 = ResNet(config='ResNet18')
#   * MobileNetV2
mnv2 = MobileNetV2(config='standard', capacity=1.0)


import torch
import quantlib.newediting as qe

##################
# F2F CONVERSION #
##################

# 1. trace graphs
gmrn18 = qe.graphs.quantlib_fine_symbolic_trace(root=rn18)
gmmnv2 = qe.graphs.quantlib_fine_symbolic_trace(root=mnv2)

# 2. create `Editor`s

# general-purpose `Editor`s
am   = qe.editing.f2f.ActivationModulariser()
lbnf = qe.editing.f2f.LinearBNBiasFolder()
# ResNet18-specific `Editor`
rn18f2fspec = [({'types': ('nn.Conv2d', 'nn.Linear')}, {'bitwidth': 8, 'signed': True},  'per-outchannel_weights', 'minmax', 'PACT'),
               ({'types': 'nn.ReLU'},                  {'bitwidth': 8, 'signed': False}, 'per-array',              ('const', {'a': 0.0, 'b': 6.0}),  'PACT')]
rn18f2f = qe.editing.f2f.F2FConverter(rn18f2fspec)
rn18converter = qe.editing.ComposedEditor([am, lbnf, rn18f2f])

# general-purpose `Editor`s
am   = qe.editing.f2f.ActivationModulariser()
lbnf = qe.editing.f2f.LinearBNBiasFolder()
# MobileNetV2-specific `Editor`
mnv2f2fspec = [({'types': ('nn.Conv2d', 'nn.Linear')}, {'bitwidth': 8, 'signed': True},  'per-outchannel_weights', 'minmax', 'PACT'),
               ({'types': ('nn.ReLU',   'nn.ReLU6')},  {'bitwidth': 8, 'signed': False}, 'per-array',              ('const', {'a': 0.0, 'b': 6.0}),  'PACT')]
mnv2f2f = qe.editing.f2f.F2FConverter(mnv2f2fspec)
mnv2converter = qe.editing.ComposedEditor([am, lbnf, mnv2f2f])

# 3. F2F-convert networks
gmrn18 = rn18converter(gmrn18)
gmmnv2 = mnv2converter(gmmnv2)

# 4. verify that the networks' semantic functionality is preserved up to this point
x = torch.randn(1, 3, 224, 224)

yrn18 = rn18(x)
ygmrn18 = gmrn18(x)
assert torch.all(ygmrn18 == yrn18)

mnv2.eval()    # MNv2 uses dropout
gmmnv2.eval()  # MNv2 uses dropout
ymnv2 = mnv2(x)
ygmmnv2 = gmmnv2(x)
assert torch.all(ygmmnv2 == ymnv2)
mnv2.train()
gmmnv2.train()

# 5. insert harmonisers

rn18harmoniserspec = {
    'algorithm':                'PACT',
    'qrangespec':               {'bitwidth': 8, 'signed': True},
    'qgranularityspec':         'per-array',
    'qhparamsinitstrategyspec': 'const',
    'force_output_scale':        True
}
rn18harmoniser = qe.editing.f2f.AddTreesHarmoniser(**rn18harmoniserspec)
gmrn18 = rn18harmoniser(gmrn18)

mnv2harmoniserspec = {
    'algorithm':                'PACT',
    'qrangespec':               {'bitwidth': 8, 'signed': True},
    'qgranularityspec':         'per-array',
    'qhparamsinitstrategyspec': 'minmax',
    'force_output_scale':        True
}
mnv2harmoniser = qe.editing.f2f.AddTreesHarmoniser(**mnv2harmoniserspec)
gmmnv2 = mnv2harmoniser(gmmnv2)

# 6. verify that the networks' semantic functionality is preserved up to this point
x = torch.randn(1, 3, 224, 224)

yrn18 = rn18(x)
ygmrn18 = gmrn18(x)
assert torch.all(ygmrn18 == yrn18)

mnv2.eval()    # MNv2 uses dropout
gmmnv2.eval()  # MNv2 uses dropout
ymnv2 = mnv2(x)
ygmmnv2 = gmmnv2(x)
assert torch.all(ygmmnv2 == ymnv2)
mnv2.train()
gmmnv2.train()

# 6. canonicalise graphs by avoiding chains of linear operations
gmrn18 = qe.graphs.quantlib_fine_symbolic_trace(root=gmrn18)
qmmnv2 = qe.graphs.quantlib_fine_symbolic_trace(root=gmmnv2)

from quantlib.newediting.editing.floattofake.interposition.linearinterposer import AllQuantiserInterposer
interposerspec = {
    'algorithm': 'PACT',
    'qrangespec': {'bitwidth': 8, 'signed': True},
    'qgranularityspec': 'per-array',
    'qhparamsinitstrategyspec': 'minmax',
}
aqi = AllQuantiserInterposer(**interposerspec)

# [features_0_residual_branch_1_1,
#  features_1_residual_branch_2_1,
#  features_3_residual_branch_2_1,
#  features_6_residual_branch_2_1,
#  features_10_residual_branch_2_1,
#  features_13_residual_branch_2_1,
#  features_16_residual_branch_2_1]

gmrn18 = aqi(gmrn18)
gmmnv2 = aqi(gmmnv2)


####################
# TRAINING/RUNTIME #
####################

import torch.fx as fx

from quantlib.newalgorithms.qmodules.qmodules.qmodules import _QModule
from quantlib.newediting.graphs.nn.harmoniser import AddTreeHarmoniser


def initialise_quantisation(g: fx.GraphModule) -> fx.GraphModule:

    for n, m in g.named_modules():
        if isinstance(m, (_QModule, AddTreeHarmoniser)):
            m.start_observing()

    for i in range(0, 10):
        x = torch.randn(4, 3, 224, 224)
        _ = g(x)  # statistics are collected by `TensorObserver`s

    for n, m in g.named_modules():
        if isinstance(m, _QModule):
            m.stop_observing()

    return g


gmrn18 = initialise_quantisation(gmrn18)
_ = gmrn18(x)

gmmnv2 = initialise_quantisation(gmmnv2)
_ = gmmnv2(x)


##################
# F2T conversion #
##################

# 1. dissolve `Harmoniser`s
gmrn18.eval()
gmrn18 = qe.graphs.quantlib_fine_symbolic_trace(root=gmrn18)

gmmnv2.eval()
gmmnv2 = qe.graphs.quantlib_fine_symbolic_trace(root=gmmnv2)

# 2. annotate the computational graphs
sp = qe.editing.f2t.ShapePropagator()
ep = qe.editing.f2t.EpsPropagator()

gmrn18 = sp.apply(gmrn18, {'x': torch.Size((1, 3, 224, 224))})
gmrn18 = ep.apply(gmrn18, {'x': 0.020625000819563866})

gmmnv2 = sp.apply(gmmnv2, {'x': torch.Size((1, 3, 224, 224))})
gmmnv2 = ep.apply(gmmnv2, {'x': 0.020625000819563866})

# 3. insert precision tunnels
eti = qe.editing.f2t.EpsTunnelInserter()
gmrn18 = eti(gmrn18)
gmmnv2 = eti(gmmnv2)

# 4. integerisation
# linear ops
# loi = qe.editing.f2t.LinearOpsIntegeriser()
from quantlib.newediting.editing.faketotrue.integerisation.linearintegeriser import AllLinOpIntegeriser
loi = AllLinOpIntegeriser()
gmrn18 = loi(gmrn18)
gmmnv2 = loi(gmmnv2)

# requant
# rq = qe.editing.f2t.Requantiser()
from quantlib.newediting.editing.faketotrue.integerisation.bnactivationrequantiser import AllRequantiser
rq = AllRequantiser()
gmrn18 = rq(gmrn18)
gmmnv2 = rq(gmmnv2)

# 5. simplify precision tunnels
from quantlib.newediting.editing.faketotrue.epstunnels.simplifier.rewriter import EpsTunnelConstructSimplifier
etcs = EpsTunnelConstructSimplifier()
gmrn18 = etcs(gmrn18)
gmmnv2 = etcs(gmmnv2)

# 6. custom rewritings
from quantlib.newediting.graphs.nn.epstunnel import EpsTunnel
import torch.nn as nn


class ResNetHead(nn.Module):

    def __init__(self):
        super(ResNetHead, self).__init__()
        self.eps_in  = EpsTunnel(torch.Tensor([1.0]))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear  = nn.Linear(1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.eps_in(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class MNv2Head(nn.Module):

    def __init__(self):
        super(MNv2Head, self).__init__()
        self.eps_in  = EpsTunnel(torch.Tensor([1.0]))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.linear  = nn.Linear(1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.eps_in(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x


from typing import List
from quantlib.newediting.editing.editors.base import ApplicationPoint
from quantlib.newediting.editing.editors.generalgraphs.rewriter import GenericGraphPattern, GenericGraphRewriter
from quantlib.newediting.graphs import quantlib_fine_symbolic_trace


class ResNetHeadRewriter(GenericGraphRewriter):

    def __init__(self):
        super(ResNetHeadRewriter, self).__init__(GenericGraphPattern(ResNetHead(), quantlib_fine_symbolic_trace, {}), 'ResNetHeadRewriter')

    def _check_aps_independence(self, aps: List[ApplicationPoint]) -> None:
        pass  # TODO: implement the check!

    def _apply(self, g: fx.GraphModule, ap: ApplicationPoint) -> fx.GraphModule:

        name_to_match_node = self._matcher.pattern.name_to_match_node(nodes_map=ap.core)
        node_linear  = name_to_match_node['linear']

        name_to_match_module = self._matcher.pattern.name_to_match_module(nodes_map=ap.core, data_gm=g)
        module_eps_in = name_to_match_module['eps_in']
        module_linear = name_to_match_module['linear']

        assert module_eps_in.eps_out.numel() == 1
        assert len(node_linear.all_input_nodes) == 1

        self._counter += 1

        new_module = nn.Linear(in_features=module_linear.in_features, out_features=module_linear.out_features, bias=module_linear.bias is not None)
        new_weight = module_linear.weight.data.detach().clone() * module_eps_in.eps_out
        new_module.weight.data = new_weight
        if module_linear.bias is not None:
            new_bias = module_linear.bias.data.detach().clone()
            new_module.bias.data = new_bias

        new_target = '_'.join([self._editor_id.upper(), new_module.__class__.__name__.upper(), str(self._counter)])

        # add the requantised linear operation to the graph...
        g.add_submodule(new_target, new_module)
        linear_input = next(iter(node_linear.all_input_nodes))
        with g.graph.inserting_after(linear_input):
            new_node = g.graph.call_module(new_target, args=(linear_input,))
        node_linear.replace_all_uses_with(new_node)

        module_eps_in.set_eps_out(torch.ones_like(module_eps_in._eps_out))  # TODO: must have the same shape as the input eps

        # ...and delete the old operation
        g.delete_submodule(node_linear.target)
        g.graph.erase_node(node_linear)

        return g


class MNv2HeadRewriter(GenericGraphRewriter):

    def __init__(self):
        super(MNv2HeadRewriter, self).__init__(GenericGraphPattern(MNv2Head(), quantlib_fine_symbolic_trace, {}), 'MNv2HeadRewriter')

    def _check_aps_independence(self, aps: List[ApplicationPoint]) -> None:
        pass  # TODO: implement the check!

    def _apply(self, g: fx.GraphModule, ap: ApplicationPoint) -> fx.GraphModule:

        name_to_match_node = self._matcher.pattern.name_to_match_node(nodes_map=ap.core)
        node_dropout = name_to_match_node['dropout']
        node_linear  = name_to_match_node['linear']

        name_to_match_module = self._matcher.pattern.name_to_match_module(nodes_map=ap.core, data_gm=g)
        module_eps_in = name_to_match_module['eps_in']
        module_linear = name_to_match_module['linear']

        assert module_eps_in.eps_out.numel() == 1
        assert len(node_dropout.all_input_nodes) == 1

        self._counter += 1

        new_module = nn.Linear(in_features=module_linear.in_features, out_features=module_linear.out_features, bias=module_linear.bias is not None)
        new_weight = module_linear.weight.data.detach().clone() * module_eps_in.eps_out
        new_module.weight.data = new_weight
        if module_linear.bias is not None:
            new_bias = module_linear.bias.data.detach().clone()
            new_module.bias.data = new_bias

        new_target = '_'.join([self._editor_id.upper(), new_module.__class__.__name__.upper(), str(self._counter)])

        # add the requantised linear operation to the graph...
        g.add_submodule(new_target, new_module)
        dropout_input = next(iter(node_dropout.all_input_nodes))
        with g.graph.inserting_after(dropout_input):
            new_node = g.graph.call_module(new_target, args=(dropout_input,))
        node_linear.replace_all_uses_with(new_node)

        module_eps_in.set_eps_out(torch.ones_like(module_eps_in._eps_out))  # TODO: must have the same shape as the input eps

        # ...and delete the old operation
        g.delete_submodule(node_linear.target)
        g.graph.erase_node(node_linear)
        g.delete_submodule(node_dropout.target)
        g.graph.erase_node(node_dropout)

        return g


rn18headr = ResNetHeadRewriter()
gmrn18 = rn18headr(gmrn18)

mnv2headr = MNv2HeadRewriter()
gmmnv2 = mnv2headr(gmmnv2)


# 7. remove precision tunnels
etr = qe.editing.f2t.EpsTunnelRemover()
gmrn18 = etr(gmrn18)
gmmnv2 = etr(gmmnv2)
