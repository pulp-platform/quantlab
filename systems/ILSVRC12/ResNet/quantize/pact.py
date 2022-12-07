# 
# pact.py
# 
# Author(s):
# Georg Rutishauser <georgr@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2021 ETH Zurich.
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


from torch import nn

import quantlib.editing.lightweight as qlw
from quantlib.editing.lightweight import LightweightGraph
import quantlib.editing.lightweight.rules as qlr
from quantlib.editing.lightweight.rules.filters import VariadicOrFilter, NameFilter, TypeFilter
from quantlib.editing.fx.passes.pact import HarmonizePACTNetPass, PACT_symbolic_trace

from quantlib.algorithms.pact.pact_ops import *
from quantlib.algorithms.pact.pact_controllers import *

def pact_recipe(net : nn.Module,
                config : dict):

    # config is expected to contain 3 keys:
    # PACTConv2d, PACTLinear, PACTUnsignedAct
    # their values are dicts with keys that will be used as NameFilter
    # arguments containing the kwargs for each layer.
    # An additional dict is expected to be stored under the key "kwargs", which
    # is used as the default kwargs.

    filter_conv2d = TypeFilter(nn.Conv2d)
    filter_linear = TypeFilter(nn.Linear)
    act_types = (nn.ReLU, nn.ReLU6)
    filter_acts = VariadicOrFilter(*[TypeFilter(t) for t in act_types])

    rhos = []
    conv_cfg = config["PACTConv2d"]
    lin_cfg = config["PACTLinear"]
    act_cfg = config["PACTUnsignedAct"]

    harmonize_cfg = config["harmonize"]

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
                       qlr.pact.ReplaceConvLinearPACTRule)
    rhos += make_rules(lin_cfg,
                       qlr.pact.ReplaceConvLinearPACTRule)
    rhos += make_rules(act_cfg,
                       qlr.pact.ReplaceActPACTRule)

    lwg = qlw.LightweightGraph(net)
    lwe = qlw.LightweightEditor(lwg)

    lwe.startup()
    for rho in rhos:
        lwe.set_lwr(rho)
        lwe.apply()
    lwe.shutdown()
    # now harmonize the graph
    harmonize_pass = HarmonizePACTNetPass(**harmonize_cfg)
    #harmonize_pass = HarmonizePACTNetPass(n_levels=harmonize_cfg["n_levels"])
    net_traced = PACT_symbolic_trace(lwg.net)
    final_net = harmonize_pass(net_traced)

    return final_net

def get_pact_controllers(net : nn.Module, schedules : dict, kwargs_linear : dict = {}, kwargs_activation : dict = {}):
    filter_fc = TypeFilter(PACTLinear)
    filter_conv = TypeFilter(PACTConv2d)
    filter_lin = filter_fc | filter_conv
    filter_act = TypeFilter(PACTUnsignedAct) | TypeFilter(PACTAsymmetricAct)
    filter_intadd = TypeFilter(PACTIntegerAdd)
    net_nodes_intadds_dissolved = LightweightGraph.build_nodes_list(net)
    net_nodes_intadds_intact = LightweightGraph.build_nodes_list(net, leaf_types=(PACTIntegerAdd,))
    lin_modules = [n.module for n in filter_lin(net_nodes_intadds_dissolved)]
    act_modules = [n.module for n in filter_act(net_nodes_intadds_dissolved)]
    intadd_modules = [n.module for n in filter_intadd(net_nodes_intadds_intact)]

    print("act_modules")
    print(act_modules)

    lin_ctrl = PACTLinearController(lin_modules, schedules["linear"], **kwargs_linear)
    act_ctrl = PACTActController(act_modules, schedules["activation"], **kwargs_activation)
    intadd_ctrl = PACTIntegerModulesController(intadd_modules)

    return lin_ctrl, act_ctrl, intadd_ctrl
