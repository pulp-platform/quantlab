from typing import Union, Optional, Literal
from functools import partial
import torch
from torch import nn
from quantlib.editing.lightweight.rules.filters import Filter, VariadicOrFilter, NameFilter, TypeFilter
import quantlib.editing.lightweight as qlw
from quantlib.editing.lightweight import LightweightGraph
from quantlib.editing.lightweight.rules import LightweightRule
from quantlib.editing.lightweight.rules.bb import ReplaceConvLinearBBRule, ReplaceActBBRule
from quantlib.algorithms.pact.pact_controllers import *
from quantlib.algorithms.bb.bb_ops import *
from quantlib.algorithms.bb.bb_controllers import *
from quantlib.editing.fx.passes.bb import *



def bb_recipe(net : nn.Module,
              config : dict,
              gate_init : float = 2.,
              joint_distribution : bool = False,
              shared_gates : bool = False,
              target : Literal["bops", "latency"] = "bops",
              latency_spec_file : Optional[str] = None,
              init_best_latency : bool = False,
              split : str = None,
              init_ctrls : bool = True):

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

    if init_ctrls:
        ctrl_pass = BBActConvControllerInitPass(shape_in=(1, 3, 224, 224), gate_init=gate_init, input_prec=8, joint_distribution=joint_distribution, shared_gates=shared_gates, target=target, latency_spec_file=latency_spec_file, init_best_latency_gates=init_best_latency, split=split)
        net_traced = BB_symbolic_trace(net)

        net_final = ctrl_pass.apply(net_traced)
        for n, m in net_final.named_modules():
            if isinstance(m, BBAct) and m.bb_gates is None:
                print(f"Activation {n} has no bb_gates!!")
    else:
        net_final = net
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

