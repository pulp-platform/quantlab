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
from functools import partial
from typing import Union

from torch import nn


from quantlib.editing.lightweight.rules.filters import Filter, VariadicOrFilter, NameFilter, TypeFilter
import quantlib.editing.lightweight as qlw
from quantlib.editing.lightweight import LightweightGraph
from quantlib.editing.lightweight.rules import LightweightRule
from quantlib.editing.lightweight.rules.bb import *
import quantlib.editing.lightweight.rules as qlr
from quantlib.editing.lightweight.rules.filters import VariadicOrFilter, NameFilter, TypeFilter
from quantlib.editing.fx.passes.pact import HarmonizePACTNetPass, PACT_symbolic_trace

from quantlib.algorithms.bb.bb_ops import *
from quantlib.algorithms.bb.bb_controllers import *
from quantlib.algorithms.pact.pact_ops import *
from quantlib.algorithms.pact.pact_controllers import *
from quantlib.editing.fx.passes.bb import *


def bb_recipe(net : nn.Module,
              config : dict,
              gate_init : float = 2.,
              shared_gates : bool = False,
              target : Literal["bops", "latency"] = "bops",
              latency_spec_file : Optional[str] = None,
              joint_distribution : bool = False
              ):

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

    filter_intadd = TypeFilter(PACTIntegerAdd)

    filter_acts = VariadicOrFilter(*[TypeFilter(t) for t in act_types])

    rhos = []
    conv_cfg = config["BBConv2d"]
    lin_cfg = config["BBLinear"]
    act_cfg = config["BBAct"]

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

    # now harmonize the graph according to the configuration
    harmonize_pass = HarmonizePACTNetPass(**harmonize_cfg)
    harmonized_net = harmonize_pass(net)

    #next, replace output activations of integerAdd modules with BB signed activations
    harmonized_nodes = LightweightGraph.build_nodes_list(harmonized_net, leaf_types=(PACTIntegerAdd,))
    intadd_modules = [n.module for n in filter_intadd(harmonized_nodes)]
    # the config for the new BB activation modules is the default activation
    # config
    bb_harmonize_act_cfg = act_cfg["kwargs"]
    bb_harmonize_act_cfg["signed"] = True
    bb_harmonize_act_cfg["act_kind"] = "identity"

    # for m in intadd_modules:
    #     m._modules['act_out'] = BBAct(**bb_harmonize_act_cfg)

    # now we can attach the controllers
    ctrl_pass = BBActConvControllerInitPass(shape_in=(1, 3, 224, 224), gate_init=gate_init, input_prec=8, joint_distribution=joint_distribution, shared_gates=shared_gates, target=target, latency_spec_file=latency_spec_file)
    net_final = ctrl_pass(harmonized_net)

    return net_final

def get_bb_controllers(net : nn.Module, schedules : dict, kwargs_linear : dict = {}, kwargs_activation : dict = {}, export_file : Optional[str] = None):
    net_nodes_intadds_dissolved = LightweightGraph.build_nodes_list(net)
    net_nodes_intadds_intact = LightweightGraph.build_nodes_list(net, leaf_types=(PACTIntegerAdd,))
    lin_modules = PACTLinearController.get_modules(net)
    act_modules = PACTActController.get_modules(net)
    intadd_modules = PACTIntegerModulesController.get_modules(net)
    lin_ctrl = PACTLinearController(lin_modules, schedules["linear"], **kwargs_linear)
    act_ctrl = PACTActController(act_modules, schedules["activation"], **kwargs_activation)
    intadd_ctrl = PACTIntegerModulesController(intadd_modules)


    if export_file is not None:
        bb_pact_filter = TypeFilter(BBLinear) | TypeFilter(BBConv2d) | TypeFilter(BBAct) | TypeFilter(PACTAsymmetricAct)
        bb_nodes = bb_pact_filter(net_nodes_intadds_dissolved)
        bb_export_ctrl = BBExportController(bb_nodes, export_file, net=net)
        return lin_ctrl, act_ctrl, intadd_ctrl, bb_export_ctrl
    else:
        return lin_ctrl, act_ctrl, intadd_ctrl
