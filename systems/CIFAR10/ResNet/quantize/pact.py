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

from torch import nn

import quantlib.editing.lightweight as qlw
from quantlib.editing.lightweight import LightweightGraph
import quantlib.editing.lightweight.rules as qlr
from quantlib.editing.lightweight.rules.filters import VariadicOrFilter, NameFilter, TypeFilter
from quantlib.editing.fx.passes.pact import HarmonizePACTNetPass, PACT_symbolic_trace
from quantlib.editing.fx.util import module_of_node

from quantlib.algorithms.pact.pact_ops import *
from quantlib.algorithms.pact.pact_controllers import *

def pact_recipe(net : nn.Module,
                config : dict,
                precision_spec_file : Optional[str] = None,
                finetuning_ckpt : Optional[str] = None):

    # config is expected to contain 3 keys:
    # PACTConv2d, PACTLinear, PACTUnsignedAct
    # their values are dicts with keys that will be used as NameFilter
    # arguments containing the kwargs for each layer.
    # An additional dict is expected to be stored under the key "kwargs", which
    # is used as the default kwargs.

    uact_types = (nn.ReLU, nn.ReLU6)
    sact_types = (nn.Hardtanh,)

    rhos = []
    conv_cfg = config["PACTConv2d"]
    lin_cfg = config["PACTLinear"]
    uact_cfg = config["PACTUnsignedAct"]

    try:
        sact_cfg = config["PACTAsymmetricAct"]
    except KeyError:
        sact_cfg = {}

    try:
        last_add_8b = config['last_add_8b']
    except KeyError:
        last_add_8b = False

    harmonize_cfg = config["harmonize"]
    
    def make_rules(cfg : dict, t : tuple,
                   rule : type, **kwargs):
        rules = []
        default_cfg = cfg["kwargs"] if "kwargs" in cfg.keys() else {}
        layer_keys = [k for k in cfg.keys() if k != "kwargs"]
        type_filter = VariadicOrFilter(*[TypeFilter(tt) for tt in t])
        print("type filter: ", type_filter)
        for k in layer_keys:
            filt = NameFilter(k) & type_filter
            kwargs.update(default_cfg)
            kwargs.update(cfg[k])
            rho = rule(filt, **kwargs)
            rules.append(rho)
        return rules

    rhos += make_rules(conv_cfg, (nn.Conv2d,),
                       qlr.pact.ReplaceConvLinearPACTRule)
    rhos += make_rules(lin_cfg, (nn.Linear,),
                       qlr.pact.ReplaceConvLinearPACTRule)
    rhos += make_rules(uact_cfg, uact_types,
                       qlr.pact.ReplaceActPACTRule, signed=False)
    rhos += make_rules(sact_cfg, sact_types,
                       qlr.pact.ReplaceActPACTRule, signed=True)

    lwg = qlw.LightweightGraph(net)
    lwe = qlw.LightweightEditor(lwg)

    lwe.startup()
    for rho in rhos:
        lwe.set_lwr(rho)
        lwe.apply()
    lwe.shutdown()    
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
        net_nodes = LightweightGraph.build_nodes_list(net)
        net_layer_names = [node.name for node in net_nodes]
        appl_keys = [k for k in prec_override_spec.keys() if k.rstrip('$') in net_layer_names]
        for k in appl_keys:
                #cfg[k]['n_levels'] = prec_override_spec[k]
            filt = NameFilter(k)
            target_module = filt(net_nodes)[0].module
            print(f"Setting module {k}'s 'n_levels' from {target_module.n_levels} to {prec_override_spec[k]}...")
            target_module.n_levels = prec_override_spec[k]

    # now harmonize the graph
    harmonize_pass = HarmonizePACTNetPass(**harmonize_cfg)
    #harmonize_pass = HarmonizePACTNetPass(n_levels=harmonize_cfg["n_levels"])
    net_traced = PACT_symbolic_trace(lwg.net)
    final_net = harmonize_pass(net_traced)


    if last_add_8b:
        for n in [nn for nn in final_net.graph.nodes][::-1]:

            if n.op == 'call_module':
                module = module_of_node(final_net, n)
                if isinstance(module, PACTIntegerAdd):
                    outact_node = [k for k in n.users.keys()][0]
                    outact_module = module_of_node(final_net, outact_node)
                    print(f"Setting node {outact_node}'s output n_levels attribute to 256!")
                    outact_module.n_levels = 256

    # the prec. spec file might include layers that were added by the
    # harmonization pass; those need to be treated separately
    final_nodes = LightweightGraph.build_nodes_list(final_net)
    for k, v in prec_override_spec.items():
        if k.startswith("_QL"):
            filt = NameFilter(k)
            target_module = filt(final_nodes)[0].module
            print(f"Setting module {k}'s 'n_levels' from {target_module.n_levels} to {v}...")
            target_module.n_levels = v

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
