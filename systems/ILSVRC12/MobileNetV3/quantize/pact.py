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

    # config is expected to contain 3 keys for each layer type:
    # PACTConv2d, PACTLinear, PACTUnsignedAct
    # their values are dicts with keys that will be used as NameFilter
    # arguments containing the kwargs for each layer.
    # An additional dict is expected to be stored under the key "kwargs", which
    # is used as the default kwargs.
    # Under the key "harmonize", the configuration for the harmonization pass
    # should be stored.

    filter_conv = TypeFilter(nn.Conv2d)
    filter_linear = TypeFilter(nn.Linear)
    relu_filter = TypeFilter(nn.ReLU)
    hswish_filter = TypeFilter(nn.Hardswish)
    hsigm_filter = TypeFilter(nn.Hardsigmoid)


    rhos = []

    def rearrange_config(cfg : dict): # move entries of "PACTUnsignedAct" whose
        # keys are also present under "PACTHardsigmoid" - same for SignedAct
        # and PACTHardswish
        new_cfg = cfg.copy()
        new_cfg["PACTHardActs"] = {}
        # 1. extract hardsigmoid/hardswish keys
        hsgm_keys = [k for k in cfg["PACTHardsigmoid"].keys() if k!="kwargs"]
        hsw_keys = [k for k in cfg["PACTHardswish"].keys() if k!="kwargs"]

        ha_cfg = {}
        for k1 in hsgm_keys:
            ha_cfg[k1] = {"hact_kwargs" : cfg["PACTHardsigmoid"]["kwargs"].copy()}
            ha_cfg[k1]["hact_kwargs"].update(cfg["PACTHardsigmoid"][k1].copy())
            ha_cfg[k1]["quant_act_kwargs"] = cfg["PACTUnsignedAct"]["kwargs"].copy()
            ha_cfg[k1]["quant_act_kwargs"].update(cfg["PACTUnsignedAct"][k1])
            del new_cfg["PACTUnsignedAct"][k1]

        for k1 in hsw_keys:
            ha_cfg[k1] = {"hact_kwargs" : cfg["PACTHardswish"]["kwargs"].copy()}
            ha_cfg[k1]["hact_kwargs"].update(cfg["PACTHardswish"][k1].copy())
            ha_cfg[k1]["quant_act_kwargs"] = cfg["PACTAsymmetricAct"]["kwargs"].copy()
            ha_cfg[k1]["quant_act_kwargs"].update(cfg["PACTAsymmetricAct"][k1])
            del new_cfg["PACTAsymmetricAct"][k1]

        new_cfg["PACTHardActs"] = ha_cfg
        del new_cfg["PACTHardsigmoid"]
        del new_cfg["PACTHardswish"]
        return new_cfg

    config = rearrange_config(config)
    conv2_cfg = config["PACTConv2d"]
    lin_cfg = config["PACTLinear"]
    uact_cfg = config["PACTUnsignedAct"]
    hact_cfg = config["PACTHardActs"]


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

    rhos += make_rules(conv2_cfg,
                       qlr.pact.ReplaceConvLinearPACTRule)
    rhos += make_rules(lin_cfg,
                       qlr.pact.ReplaceConvLinearPACTRule)
    rhos += make_rules(uact_cfg,
                       qlr.pact.ReplaceActPACTRule)
    rhos += make_rules(hact_cfg,
                       qlr.pact.ReplaceHardActPACTRule)

    lwg = qlw.LightweightGraph(net)
    lwe = qlw.LightweightEditor(lwg)

    lwe.startup()
    for rho in rhos:
        lwe.set_lwr(rho)
        lwe.apply()
    lwe.shutdown()

    # now harmonize the graph according to the configuration
    harmonize_pass = HarmonizePACTNetPass(**harmonize_cfg)
    final_net = harmonize_pass(net)

    return final_net

def get_pact_controllers(net : nn.Module, schedules : dict, kwargs_linear : dict = {}, kwargs_activation : dict = {}):
    filter_intadd = TypeFilter(PACTIntegerAdd)
    net_nodes_intadds_dissolved = LightweightGraph.build_nodes_list(net)
    net_nodes_intadds_intact = LightweightGraph.build_nodes_list(net, leaf_types=(PACTIntegerAdd,))
    lin_modules = PACTLinearController.get_modules(net)
    act_modules = PACTActController.get_modules(net)
    intadd_modules = PACTIntegerModulesController.get_modules(net)

    lin_ctrl = PACTLinearController(lin_modules, schedules["linear"], **kwargs_linear)
    act_ctrl = PACTActController(act_modules, schedules["activation"], **kwargs_activation)
    intadd_ctrl = PACTIntegerModulesController(intadd_modules)

    return lin_ctrl, act_ctrl, intadd_ctrl
