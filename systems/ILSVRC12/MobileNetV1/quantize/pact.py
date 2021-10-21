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

from copy import deepcopy

from torch import nn

import quantlib.editing.lightweight as qlw
from quantlib.editing.lightweight import LightweightGraph
import quantlib.editing.lightweight.rules as qlr
from quantlib.editing.lightweight.rules.filters import VariadicOrFilter, NameFilter, TypeFilter
from quantlib.editing.fx.passes.pact import HarmonizePACTNetPass, PACT_symbolic_trace

from quantlib.algorithms.pact.pact_ops import *
from quantlib.algorithms.pact.pact_controllers import *
from .precision_search import quant_layers_from_net, cut_acts_wts_orig, cut_acts_wts_pulp, cut_acts_wts_pulp_mp, print_net_summary, QuantConvLayer, QuantLinLayer

def config_from_layers(ql : list, nodes_list : list):
    # from a list of quantConv/quantLinear and the complete list of nodes
    # representing a sequential net, return a config for PACTSequential
    cfg = {'PACTConv2d':{}, 'PACTLinear':{}, 'PACTUnsignedAct':{}}

    for l in ql:
        if isinstance(l, QuantConvLayer):
            key = 'PACTConv2d'
        else:
            key = 'PACTLinear'
        cfg[key][l.node.name+'$'] = {'n_levels' : int(2**l.q_wt)}

        # search for the next activation layer in the network
        i = nodes_list.index(l.node)
        for ll in nodes_list[i:]:
            if isinstance(ll.module, nn.ReLU):
                cfg['PACTUnsignedAct'][ll.name+'$'] = {'n_levels' : int(2**l.q_out)}
                break
    return cfg


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
    # memory-constrained mixed-precision configuration search
    try:
        mp_search = config['mixed_precision_search']['enable']
    except KeyError:
        mp_search = False

    if mp_search:
        # avoid polluting the config somewhere else in the quantlab flow
        config = deepcopy(config)

        mp_config = config['mixed_precision_search']
        qa_min = mp_config['min_act_bits']
        qw_min = mp_config['min_wt_bits']
        method = mp_config['method']
        in_dims = (mp_config['in_size'], ) * 2

        mem_rw = mp_config['mem_rw']

        lin_split = mp_config['linear_layer_split_factor']

        ql = quant_layers_from_net(net, in_dims, last_layer_split=lin_split)
        if method == 'pulp':
            if qa_min != qw_min:
                print(f"WARNING: Mixed precision search method 'pulp' enforces identical precisions for activations & weights - you supplied differing values. Using maximum ({max(qa_min, qw_min)} bits) as shared minimum precision.")
            print("Cutting weights & activation precisions using homogeneous-precision (weights and activations must have the same precision) method targeting PULP systems")
            ql_cut = cut_acts_wts_pulp(ql, mem_rw, min(qw_min, qa_min), qw_min)
        elif method == 'pulp_mp':
            print("Cutting weights & activation precisions using mixed-precision (weights and activations may have different precisions) method targeting PULP systems")
            ql_cut = cut_acts_wts_pulp_mp(ql, mem_rw, qw_min, qa_min)
        else:
            assert method  == 'orig', f"Unknown precision search method {method}!"
            print("Cutting weight & activation precisions using original method from paper 'Memory-Driven Mixed Precision Quantization'")
            mem_ro = mp_config['mem_ro']
            delta = mp_config['delta']
            ql_cut = cut_acts_wts_orig(ql, mem_ro, mem_rw, qw_min, qa_min, delta)

        print_net_summary(ql_cut)
        nodes_list = LightweightGraph.build_nodes_list(net)
        quant_cfg = config_from_layers(ql_cut, nodes_list)
        # iterate over 'PACTConv2d', 'PACTLinear', ...
        for k in quant_cfg:
            # iterate over 'pilot.0', 'features.0.2', ...
            for lk in quant_cfg[k]:
                if lk in config[k]:
                    config[k][lk].update(quant_cfg[k][lk])
                else:
                    config[k][lk] = quant_cfg[k][lk].copy()

    filter_conv2d = TypeFilter(nn.Conv2d)
    filter_linear = TypeFilter(nn.Linear)
    act_types = (nn.ReLU, nn.ReLU6)
    filter_acts = VariadicOrFilter(*[TypeFilter(t) for t in act_types])

    rhos = []
    conv_cfg = config["PACTConv2d"]
    lin_cfg = config["PACTLinear"]
    act_cfg = config["PACTUnsignedAct"]

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

    return lwe._graph.net

def get_pact_controllers(net : nn.Module, schedules : dict, kwargs_linear : dict = {}, kwargs_activation : dict = {}):
    filter_intadd = TypeFilter(PACTIntegerAdd)
    net_nodes_intadds_dissolved = LightweightGraph.build_nodes_list(net)
    net_nodes_intadds_intact = LightweightGraph.build_nodes_list(net, leaf_types=(PACTIntegerAdd,))
    lin_modules = PACTLinearController.get_modules(net)
    act_modules = PACTActController.get_modules(net)

    lin_ctrl = PACTLinearController(lin_modules, schedules['linear'], **kwargs_linear)
    act_ctrl = PACTActController(act_modules, schedules['activation'], **kwargs_activation)

    return lin_ctrl, act_ctrl
