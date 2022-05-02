from typing import Union, Optional
from functools import partial
import torch
from torch import nn
from quantlib.editing.lightweight.rules.filters import Filter, VariadicOrFilter, NameFilter, TypeFilter
import quantlib.editing.lightweight as qlw
from quantlib.editing.lightweight import LightweightGraph
from quantlib.editing.lightweight.rules import LightweightRule
from quantlib.algorithms.pact.pact_controllers import *
from quantlib.algorithms.bb.bb_ops import *
from quantlib.algorithms.bb.bb_controllers import *
from quantlib.editing.fx.passes.bb import *

class ReplaceConvLinearBBRule(LightweightRule):
    @staticmethod
    def replace_bb_conv_linear(module : Union[nn.Conv2d, nn.Linear], **kwargs):
        if isinstance(module, nn.Conv2d):
            return BBConv2d.from_conv2d(module, **kwargs)
        elif isinstance(module, nn.Linear):
            return BBLinear.from_linear(module, **kwargs)
        else:
            raise TypeError(f"Incompatible module of type {module.__class__.__name__} passed to replace_bb_conv_linear!")

    def __init__(self,
                 filter_ : Filter,
                 **kwargs):
        replacement_fun = partial(self.replace_bb_conv_linear, **kwargs)
        super(ReplaceConvLinearBBRule, self).__init__(filter_=filter_, replacement_fun=replacement_fun)


class ReplaceActBBRule(LightweightRule):
    @staticmethod
    def replace_bb_act(module : nn.Module,
                       **kwargs):
        if 'act_kind' not in kwargs.keys():
            if isinstance(module, nn.ReLU6):
                act_kind = 'relu6'
            elif isinstance(module, nn.LeakyReLU):
                act_kind = 'leaky_relu'
                if 'leaky' not in kwargs:
                    kwargs['leaky'] = module.negative_slope
            else: # default activation is ReLU
                act_kind = 'relu'

            kwargs['act_kind'] = act_kind
        return BBAct(**kwargs)

    def __init__(self,
                 filter_ : Filter,
                 **kwargs):
        replacement_fun = partial(self.replace_bb_act, **kwargs)
        super(ReplaceActBBRule, self).__init__(filter_=filter_, replacement_fun=replacement_fun)


def bb_recipe(net : nn.Module,
              config : dict,
              gate_init : float = 2.,
              joint_distribution : bool = False,
              shared_gates : bool = False):

    assert not (shared_gates and joint_distribution), "shared_gates and joint_distribution are mutually exclusive!"
    filter_conv2d = TypeFilter(nn.Conv2d)
    filter_linear = TypeFilter(nn.Linear)
    act_types = (nn.ReLU, nn.ReLU6)
    filter_acts = VariadicOrFilter(*[TypeFilter(t) for t in act_types])
    rhos = []
    conv_cfg = config["BBConv2d"]
    lin_cfg = config["BBLinear"]
    act_cfg = config["BBAct"]


    def make_rules(cfg : dict,
                   rule : type):
        rules = []
        default_cfg = cfg["kwargs"] if "kwargs" in cfg.keys() else {}
        layer_keys = [k for k in cfg.keys() if k != "kwargs"]
        for k in layer_keys:
            filt = NameFilter(k)
            kwargs = default_cfg.copy()
            kwargs.update(cfg[k])
            rho = rule(filt, **kwargs)
            rules.append(rho)
        return rules

    rhos += make_rules(conv_cfg,
                       ReplaceConvLinearBBRule)
    rhos += make_rules(lin_cfg,
                       ReplaceConvLinearBBRule)
    rhos += make_rules(act_cfg,
                       ReplaceActBBRule)

    lwg = qlw.LightweightGraph(net)
    lwe = qlw.LightweightEditor(lwg)

    lwe.startup()
    for rho in rhos:
        lwe.set_lwr(rho)
        lwe.apply()
    lwe.shutdown()

    net = lwe._graph.net

    #attach gate controllers using the appropriate pass
    #ctrl_pass = BBControllerInitPass(shape_in=(1, 3, 224, 224),
    #gate_init=gate_init)
    ctrl_pass = BBActConvControllerInitPass(shape_in=(1, 3, 224, 224), gate_init=gate_init, input_prec=8, joint_distribution=joint_distribution, shared_gates=shared_gates)
    net_traced = BB_symbolic_trace(net)

    net_final = ctrl_pass.apply(net_traced)
    for n, m in net_final.named_modules():
        if isinstance(m, BBAct) and m.bb_gates is None:
            print(f"Activation {n} has no bb_gates!!")
    return net_final


def get_bb_controllers(net : nn.Module, schedules : dict, kwargs_linear : dict = {}, kwargs_activation : dict = {}, export_file : Optional[str] = None):
    net_nodes = LightweightGraph.build_nodes_list(net)
    lin_filter = TypeFilter(BBLinear) | TypeFilter(BBConv2d)
    act_filter = TypeFilter(BBAct)
    bb_filter = lin_filter | act_filter
    lin_modules = [n.module for n in lin_filter(net_nodes)]
    act_modules = [n.module for n in act_filter(net_nodes)]


    lin_ctrl = PACTLinearController(lin_modules, schedules['linear'], **kwargs_linear)
    act_ctrl = PACTActController(act_modules, schedules['activation'], **kwargs_activation)

    bb_nodes = bb_filter(net_nodes)

    if export_file is not None:
        bb_export_ctrl = BBExportController(bb_nodes, export_file, input_bitwidth=8, net=net)
        return lin_ctrl, act_ctrl, bb_export_ctrl
    else:
        return lin_ctrl, act_ctrl

