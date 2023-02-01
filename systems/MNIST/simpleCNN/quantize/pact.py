# 
# pact.py
# 
# Author(s):
# Philip Wiese <wiesep@student.ethz.ch>
# 
# Copyright (c) 2023 ETH Zurich.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

from typing import List

from torch import nn

import quantlib.editing.lightweight as qlw
from quantlib.editing.lightweight import LightweightGraph
import quantlib.editing.lightweight.rules as qlr
from quantlib.editing.lightweight.rules.filters import VariadicOrFilter, NameFilter, TypeFilter, SubTypeFilter
from quantlib.editing.fx.passes.pact import HarmonizePACTNetPass, PACT_symbolic_trace, LeafTracer, PACT_OPS_INCLUSIVE
from quantlib.editing.fx.passes import AnnotateEpsPass

from quantlib.algorithms.pact.pact_ops import *
from quantlib.algorithms.pact.pact_controllers import *
from quantlib.algorithms.pact.dynamic_precision import *




def pact_recipe(net : nn.Module,
                config : dict):

    # config is expected to contain 3 keys for each layer type:
    # PACTConv2d, PACTLinear, PACTUnsignedAct
    # their values are dicts with keys that will be used as NameFilter
    # arguments containing the kwargs for each layer.
    # An additional dict is expected to be stored under the key "kwargs", which
    # is used as the default kwargs.
    # Under the key "harmonize", the configuration for the harmonization pass
    # should be stored.

    filter_conv2d = TypeFilter(nn.Conv2d)
    filter_linear = TypeFilter(nn.Linear)
    filter_acts = TypeFilter(nn.ReLU)
    # filter_softmax = TypeFilter(nn.LogSoftmax) #WIESEP: Not sure if this works

    rhos = []
    conv_cfg = config["PACTConv2d"]
    lin_cfg = config["PACTLinear"]
    act_cfg = config["PACTUnsignedAct"]
    # softmax_cfg = config["PACTIntegerSoftmax"]


    def make_rules(cfg : dict,
                   rule : type):
        rules = []
        default_cfg = cfg["kwargs"] if "kwargs" in cfg.keys() else {}
        layer_keys = [k for k in cfg.keys() if k != "kwargs" and k != "dynamic"]
        for k in layer_keys:
            filt = NameFilter(k)
            kwargs = default_cfg.copy()
            kwargs.update(cfg[k])
            rho = rule(filt, **kwargs)
            rules.append(rho)
        return rules

    rhos += make_rules(conv_cfg,
                       qlr.pact.ReplaceConvLinearPACTRule)
    rhos += make_rules(lin_cfg,
                       qlr.pact.ReplaceConvLinearPACTRule)
    rhos += make_rules(act_cfg,
                       qlr.pact.ReplaceActPACTRule)

    lwg = qlw.LightweightGraph(net)
    lwe = qlw.LightweightEditor(lwg)
    
    print("=== Original Network ===")
    lwg.show_nodes_list()
    
    lwe.startup()
    for rho in rhos:
        lwe.set_lwr(rho)
        lwe.apply()
    lwe.shutdown()

    print("=== PACT Network ===")
    lwg.show_nodes_list()

    return lwe._graph.net

def build_module_spec(net : nn.Module, names : List[str], cfg : dict):
    select_levels_dict = {"static": select_levels_static,
                          "uniform": select_levels_uniform,
                          "const": select_levels_const,
                          "anneal": select_levels_anneal}

    modules = [net.get_submodule(name) for name in names]
    levels = cfg["levels"]
    select_levels_val = select_levels_dict[cfg["select_levels_val"]](**cfg["val_kwargs"])
    select_levels_trn = select_levels_dict[cfg["select_levels_trn"]](**cfg["trn_kwargs"])
    return (modules, levels, select_levels_trn, select_levels_val)

def get_pact_controllers(net : nn.Module, schedules : dict, kwargs_linear : dict = {}, kwargs_activation : dict = {}, dynamic : dict = {}):
    net_nodes = LightweightGraph.build_nodes_list(net)
    lin_modules = PACTLinearController.get_modules(net)
    act_modules = PACTActController.get_modules(net)

    net_nodes = LightweightGraph.build_nodes_list(net)
    filter_softmax = TypeFilter(PACTSoftmax)
    # softmax_modules =  [n.module for n in filter_softmax(net_nodes)]

    lin_ctrl = PACTLinearController(lin_modules, schedules["linear"], **kwargs_linear)
    act_ctrl = PACTActController(act_modules, schedules["activation"], **kwargs_activation)
    tracer = LeafTracer(PACT_OPS_INCLUSIVE)
    # softmax_ctrl = PACTEpsController(net, softmax_modules, schedules["softmax"], tracer, AnnotateEpsPass(eps_in=1.0, n_levels_in=256))
    try:
        enable_dynamic = dynamic["cfg"]["enable"]
    except KeyError:
        enable_dynamic = False

    if enable_dynamic:
        module_spec_list = []
        module_cfgs = {k : v for k, v in dynamic.items() if k != "cfg"}
        for mn, c in module_cfgs.items():
            cfg = dynamic["cfg"].copy()
            if "layers" in c.keys():
                names = c["layers"]
                try:
                    config = c["cfg"]
                except KeyError:
                    config = {}
            else:
                names = [mn]
                config = c
            cfg.update(config)
            module_spec_list.append(build_module_spec(net, names, cfg))

        dyn_ctrl = PACTDynamicPrecController(module_spec_list)
        #######################
        # WATCH OUT!!!!!
        # The order in which these controllers are returned matters!!!
        # the dyn_ctrl must come first, because the step_xx functions are
        # called in order on each controller!
        ########################
        return dyn_ctrl, lin_ctrl, act_ctrl

    return lin_ctrl, act_ctrl
