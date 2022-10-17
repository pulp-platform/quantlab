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

import json
from typing import Optional

import torch
from torch import nn

import quantlib.editing.lightweight as qlw
from quantlib.editing.lightweight import LightweightGraph
import quantlib.editing.lightweight.rules as qlr
from quantlib.editing.lightweight.rules import LightweightRule
from quantlib.editing.lightweight.rules.filters import VariadicOrFilter, NameFilter, TypeFilter
from quantlib.editing.fx.passes.pact import HarmonizePACTNetPass, PACT_symbolic_trace

from quantlib.algorithms.pact.pact_ops import *
from quantlib.algorithms.pact.pact_controllers import *

def pact_recipe(net : nn.Module,
                config : dict,
                precision_spec_file : Optional[str] = None,
                finetuning_ckpt : Optional[str] = None,
                quantize_pool : bool = False):


    if quantize_pool:
        # in MNv1 we know the pooling node will be an adaptive avg pool
        print("Quantizing Average Pooling layer...")
        pool_filter = TypeFilter(nn.AdaptiveAvgPool2d)
        pool_nodes = pool_filter(LightweightGraph.build_nodes_list(net))
        act = nn.ReLU6
        for n in pool_nodes:
            print(f"Found pooling layer: {n.name}\nReplacing with Pool+ReLU6")
            activated_pool = nn.Sequential(n.module, act(inplace=True))
            LightweightRule.replace_module(net, n.name.split('.'), activated_pool)
            if n.name+'.1' not in config["PACTUnsignedAct"].keys():
                config["PACTUnsignedAct"][n.name+'.1'] = {}

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
    act_types = (nn.ReLU, nn.ReLU6)
    filter_acts = VariadicOrFilter(*[TypeFilter(t) for t in act_types])

    rhos = []
    conv_cfg = config["PACTConv2d"]
    lin_cfg = config["PACTLinear"]
    act_cfg = config["PACTUnsignedAct"]

    harmonize_cfg = config["harmonize"]


    prec_override_spec = {}
     # the precision_spec_file is (for example) dumped by a Bayesian Bits
    # training run and overrides the 'n_levels' spec from config.json
    if precision_spec_file is not None:
        print(f"Overriding precision specification from config.json with spec from <{precision_spec_file}>...")
        with open(precision_spec_file, 'r') as fh:
            prec_override_spec = json.load(fh)['layer_levels']
        # deal with nn.DataParallel wrapping

        if all(k.startswith('module.') for k in prec_override_spec.keys()):
            prec_override_spec = {k.lstrip('module.'):v for k,v in prec_override_spec.items()}
        for cfg in (conv_cfg, lin_cfg, act_cfg):
            appl_keys = [k for k in prec_override_spec.keys() if k in cfg.keys()]
            for k in appl_keys:
                cfg[k]['n_levels'] = prec_override_spec[k]


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

    # now harmonize the graph according to the configuration
    harmonize_pass = HarmonizePACTNetPass(**harmonize_cfg)
    final_net = harmonize_pass(net)

    # the prec. spec file might include layers that were added by the
    # harmonization pass; those need to be treated separately
    final_nodes = LightweightGraph.build_nodes_list(final_net)
    for k, v in prec_override_spec.items():
        if k.startswith("_QL"):
            filt = NameFilter(k)
            target_module = filt(final_nodes)[0].module
            print(f"Setting module {k}'s 'n_levels' from {target_module.n_levels} to {v}...")
            target_module.n_levels = v


    if finetuning_ckpt is not None:
        print(f"Loading finetuning ckpt from <{finetuning_ckpt}>...")
        state_dict = torch.load(finetuning_ckpt)['net']
        if all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.lstrip('module.'):v for k,v in state_dict.items()}
        final_net.load_state_dict(state_dict, strict=False)


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
