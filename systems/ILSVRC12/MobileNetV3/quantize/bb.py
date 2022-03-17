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
from typing import Optional

import torch
from torch import nn

import operator

import quantlib.editing.lightweight as qlw
from quantlib.editing.lightweight import LightweightGraph
import quantlib.editing.lightweight.rules as qlr

from quantlib.editing.lightweight.rules.bb import *
from quantlib.editing.lightweight.rules.filters import Filter, VariadicOrFilter, NameFilter, TypeFilter, SubTypeFilter

from quantlib.editing.fx.passes.pact import HarmonizePACTNetPass, PACT_symbolic_trace, PACT_symbolic_trace_inclusive, OpTreeReplacementPass, InsertActivationsBetweenLinearsPass, MulReplacementPass, AddTreeReplacementPass
from quantlib.editing.fx.passes import ModifySequentialPatternPass, SequentialPass, RetracePass
from quantlib.editing.fx.passes.bb import *

from quantlib.algorithms.pact.pact_ops import *
from quantlib.algorithms.bb.bb_ops import *
from quantlib.algorithms.pact.pact_controllers import *
from quantlib.algorithms.bb.bb_controllers import *



class ReplaceHardActBBRule(qlr.LightweightRule):
    def __init__(self,
                 filter_ : Filter,
                 bb_act_kwargs : dict):
        replacement_fun = partial(self.bb_quant_hard_act, bb_act_kwargs=bb_act_kwargs)
        super(ReplaceHardActBBRule, self).__init__(filter_=filter_, replacement_fun=replacement_fun)

    @staticmethod
    def bb_quant_hard_act(module : nn.Module, bb_act_kwargs : dict):
        if isinstance(module, nn.Hardswish):
            if "signed" not in bb_act_kwargs:
                bb_act_kwargs["signed"] = True
            layers = [nn.Hardswish(), BBAct(**bb_act_kwargs)]
        elif isinstance(module, nn.Hardsigmoid):
            if "signed" not in bb_act_kwargs:
                bb_act_kwargs["signed"] = False
            layers = [nn.Hardsigmoid(), BBAct(**bb_act_kwargs)]
        return nn.Sequential(*layers)

class BBAddTreeReplacementPass(OpTreeReplacementPass):
    add_node_specs = [('call_function', (torch.add, operator.add)),
                      ('call_method', ('add',))]
    def __init__(self, signed : bool, bb_kwargs : dict = {}, pact_kwargs : dict = {}):
        pact_default_kwargs = {'learn_clip' : True, 'tqt' : True, 'init_clip' : 'max', 'act_kind' : 'identity'}
        pact_default_kwargs.update(pact_kwargs)
        bb_default_kwargs = {'hc_stretch' : 1.2, 'hc_T' : 0.5, 'learn_clip' : False, 'init_clip' : 'max'}
        bb_default_kwargs.update(bb_kwargs)
        self.pact_args = pact_default_kwargs
        self.bb_args = bb_default_kwargs
        self.signed = signed
        super(BBAddTreeReplacementPass, self).__init__(node_specs=self.add_node_specs, replacement_fn=self.add_replacement_fn, name="BB_ADDITION")

    def add_replacement_fn(self, tree):
        n_args = len(tree.args)
        return BBIntegerAdd(num_args=n_args, signed=self.signed, pact_kwargs=self.pact_args, bb_args=self.bb_args)

class AdvancedInsertActivationsBetweenLinearsPass(InsertActivationsBetweenLinearsPass):
    def __init__(self, signed : bool, strategy : list, default_act_type : str, pact_kwargs : dict, bb_kwargs : dict):
        # 'strategy' should be a list of ((before_module, after_module),
        # act_type) where act_type is either "pact" or "bb"
        name = "ADVANCED_LINEAR_ACTIVATIONS"
        self.signed = signed
        bb_default_kwargs = {'hc_stretch' : 1.2, 'hc_T' : 0.5, 'learn_clip' : False, 'init_clip' : 'max', 'act_kind' : 'identity'}
        bb_default_kwargs.update(bb_kwargs)
        if 'signed' in bb_default_kwargs.keys():
            del bb_default_kwargs['signed']
        pact_default_kwargs = {'learn_clip' : True, 'tqt' : True, 'init_clip' : 'max', 'act_kind' : 'identity'}
        pact_default_kwargs.update(pact_kwargs)
        self.pact_kwargs = pact_default_kwargs
        self.bb_kwargs = bb_default_kwargs
        self.strategy = strategy
        self.default_act_type = default_act_type
        super(AdvancedInsertActivationsBetweenLinearsPass, self).__init__(modules_before=self.before_modules,
                                                                  modules_after=self.after_modules,
                                                                  make_module_fn=self.inserted_module,
                                                                  name=name,
                                                                  combine='force')

    def inserted_module(self, module_before, module_after):
        act_type = self.default_act_type
        for (tb, ta), t in self.strategy:
            if isinstance(module_before, tb) and isinstance(module_after, ta):
                act_type = t
                break
        if act_type == "pact":
            if self.signed:
                return PACTAsymmetricAct(**self.pact_kwargs)
            else:
                module_kwargs = {k:v for k, v in self.pact_kwargs.items() if k != "symm"}
                return PACTUnsignedAct(**module_kwargs)

        elif act_type == "bb":
            return BBAct(signed=self.signed, **self.bb_kwargs)
        else:
            assert False, f"AdvancedInsertActivationsBetweenLinearsPass got invalid 'act_type': {act_type}"


class HarmonizeBBNetPass(SequentialPass):
    def __init__(self, strategy : str, bb_kwargs : dict, pact_kwargs : dict):
        assert strategy in ['conservative', 'aggressive'], f"Invalid strategy for HarmonizeBBNetPass: {strategy}; expected 'conservative' or 'aggressive'"
        passes = []
        passes.append(RetracePass(BB_symbolic_trace))
        if strategy == 'conservative':
            passes.append(AddTreeReplacementPass(signed=True, **pact_kwargs))
        else:
            passes.append(BBAddTreeReplacementPass(signed=True, bb_kwargs=bb_kwargs, pact_kwargs=pact_kwargs))
        passes.append(MulReplacementPass())
        pact_actpass_kwargs = {k:v for k,v in pact_kwargs.items() if k != 'force_out_eps'}
        strat_conservative = [(((nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d), (nn.Conv1d, nn.Conv2d, nn.Conv3d)), 'pact')]
        # aggressive strategy: insert BB activations everywhere - no strategy
        # spec needed
        strat_aggressive = []
        if strategy == 'conservative':
            passes.append(AdvancedInsertActivationsBetweenLinearsPass(signed=True, strategy=strat_conservative, default_act_type='bb', pact_kwargs=pact_actpass_kwargs, bb_kwargs=bb_kwargs))
        else:
            passes.append(AdvancedInsertActivationsBetweenLinearsPass(signed=True, strategy=strat_aggressive, default_act_type='bb', pact_kwargs=pact_actpass_kwargs, bb_kwargs=bb_kwargs))
        super(HarmonizeBBNetPass, self).__init__(*passes, name_prefix='_HARMONIZE_BB_NET_PASS')


def bb_recipe(net : nn.Module,
              strategy : str,
              config : dict,
              gate_init : float,
              joint_distribution : bool):

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

    net_nodes = LightweightGraph.build_nodes_list(net)

    rhos = []

    def rearrange_config(cfg : dict): # move entries of "PACTUnsignedAct" whose
        # keys are also present under "PACTHardsigmoid" - same for SignedAct
        # and PACTHardswish
        new_cfg = cfg.copy()
        new_cfg["HardActsPACT"] = {}
        new_cfg["HardActsBB"] = {}
        # 1. extract hardsigmoid/hardswish keys
        # note: config.json must have keys ending with $ for correct regex matching!!
        hsgm_keys = [n.name+"$" for n in hsigm_filter(net_nodes)]
        hsw_keys = [n.name+"$" for n in hswish_filter(net_nodes)]
        ha_pact_cfg = {}
        ha_bb_cfg = {}
        for k1 in hsgm_keys:
            # #aggressive strategy: Use BB activations after HardSigmoid
            # if strategy == 'aggressive':
            #     ha_bb_cfg[k1] = {"bb_act_kwargs" :  cfg["BBAct"]["kwargs"].copy()}
            #     ha_bb_cfg[k1]["bb_act_kwargs"].update(cfg["BBAct"][k1])
            #     ha_bb_cfg[k1]["bb_act_kwargs"]["signed"] = False
            # #conservative strategy: PACT activations after HardSigmoid
            # else:
            #     ha_pact_cfg[k1] = {"quant_act_kwargs" : cfg["PACTUnsignedAct"]["kwargs"].copy()}
            #     ha_pact_cfg[k1]["quant_act_kwargs"].update(cfg["PACTUnsignedAct"][k1])

            # hsigm always get PACT for now - contribution of squeeze-excite
            # multiplication to compute load is insignificant
            ha_pact_cfg[k1] = {"quant_act_kwargs" : cfg["PACTUnsignedAct"]["kwargs"].copy()}
            ha_pact_cfg[k1]["quant_act_kwargs"].update(cfg["PACTUnsignedAct"][k1])
            ha_pact_cfg[k1]["use_pact_hact"] = False
            del new_cfg["PACTUnsignedAct"][k1]

        for k1 in hsw_keys:
            # HardSwishes always get a BB activation
            ha_bb_cfg[k1] = {"bb_act_kwargs" : cfg["BBAct"]["kwargs"].copy()}
            ha_bb_cfg[k1]["bb_act_kwargs"].update(cfg["BBAct"][k1])
            ha_bb_cfg[k1]["bb_act_kwargs"]["signed"] = True
            del new_cfg["BBAct"][k1]

        new_cfg["HardActsPACT"] = ha_pact_cfg
        new_cfg["HardActsBB"] = ha_bb_cfg
        return new_cfg

    config = rearrange_config(config)
    conv2_cfg = config["BBConv2d"]
    lin_cfg = config["BBLinear"]
    act_cfg = config["BBAct"]

    hact_pact_cfg = config["HardActsPACT"]
    
    hact_bb_cfg = config["HardActsBB"]

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
                       qlr.bb.ReplaceConvLinearBBRule)
    rhos += make_rules(lin_cfg,
                       qlr.bb.ReplaceConvLinearBBRule)
    rhos += make_rules(act_cfg,
                       qlr.bb.ReplaceActBBRule)
    rhos += make_rules(hact_pact_cfg,
                       qlr.pact.ReplaceHardActPACTRule)
    rhos += make_rules(hact_bb_cfg,
                       ReplaceHardActBBRule)

    lwg = qlw.LightweightGraph(net)
    lwe = qlw.LightweightEditor(lwg)

    lwe.startup()
    for rho in rhos:
        lwe.set_lwr(rho)
        lwe.apply()
    lwe.shutdown()

    # now harmonize the graph according to the configuration
    harmonize_pass = HarmonizeBBNetPass(strategy=strategy, **harmonize_cfg)
    final_net = harmonize_pass(net)


    ctrl_pass = BBActConvControllerInitPass(shape_in=(1, 3, 224, 224), gate_init=gate_init, input_prec=8, joint_distribution=joint_distribution)
    final_net = ctrl_pass(final_net)

    # change adder input activations' n_levels if configured
    if "adder_levels" in config.keys() and strategy == 'conservative':
        change_adder_levels(final_net, **config["adder_levels"])

    return final_net

def get_bb_controllers(net : nn.Module, schedules : dict, kwargs_linear : dict = {}, kwargs_activation : dict = {}, export_file : Optional[str] = None):
    filter_intadd = SubTypeFilter(PACTIntegerAdd)
    net_nodes_intadds_dissolved = LightweightGraph.build_nodes_list(net)
    net_nodes_intadds_intact = LightweightGraph.build_nodes_list(net, leaf_types=(PACTIntegerAdd, BBIntegerAdd))
    lin_modules = PACTLinearController.get_modules(net)
    act_modules = PACTActController.get_modules(net)
    intadd_modules = PACTIntegerModulesController.get_modules(net)

    lin_ctrl = PACTLinearController(lin_modules, schedules["linear"], **kwargs_linear)
    act_ctrl = PACTActController(act_modules, schedules["activation"], **kwargs_activation)
    intadd_ctrl = PACTIntegerModulesController(intadd_modules)

    if export_file is not None:
        bb_pact_filter = TypeFilter(BBLinear) | TypeFilter(BBConv2d) | TypeFilter(BBAct) | TypeFilter(PACTAsymmetricAct)
        bb_nodes = bb_pact_filter(net_nodes_intadds_dissolved)
        bb_export_ctrl = BBExportController(bb_nodes, export_file, input_bitwidth=8, net=net)
        return lin_ctrl, act_ctrl, intadd_ctrl, bb_export_ctrl

    return lin_ctrl, act_ctrl, intadd_ctrl
