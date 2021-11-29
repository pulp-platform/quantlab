#
# pact.py

# Author(s):
# Georg Rutishauser <georgr@iis.ee.ethz.ch>

# Copyright (c) 2020-2021 ETH Zurich. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from torch import nn

from quantlib.algorithms.pact import PACTUnsignedAct, PACTAsymmetricAct, PACTConv1d, PACTConv2d, PACTLinear
from quantlib.algorithms.pact import PACTActController, PACTLinearController
import quantlib.editing.lightweight as qlw
from quantlib.editing.lightweight import LightweightGraph, LightweightEditor
import quantlib.editing.lightweight.rules as qlr
from quantlib.editing.lightweight.rules import LightweightRule
from quantlib.editing.lightweight.rules.filters import TypeFilter, VariadicOrFilter, NameFilter


__all__ = ['pact_recipe',
           'get_pact_controllers']

def pact_recipe(net : nn.Module,
                config : dict):

    # config is expected to contain 3 keys:
    # PACTConv2d, PACTLinear, PACTUnsignedAct
    # their values are dicts with keys that will be used as NameFilter
    # arguments containing the kwargs for each layer.
    # An additional dict is expected to be stored under the key "kwargs", which
    # is used as the default kwargs.

    filter_convs = TypeFilter(nn.Conv2d) | TypeFilter(nn.Conv1d)
    filter_htanh = TypeFilter(nn.Hardtanh)
    filter_relu = TypeFilter(nn.ReLU) | TypeFilter(nn.ReLU6)

    rhos = []
    conv_kwargs = config["PACTConvNd"]
    signed_act_kwargs = config["PACTAsymmetricAct"]
    unsigned_act_kwargs = config["PACTUnsignedAct"]

    def make_rule(cfg : dict,
                   rule : type,
                   filt: TypeFilter):
        kwargs = cfg["kwargs"] if "kwargs" in cfg.keys() else {}
        rho = rule(filt, **kwargs)
        return rho

    rhos.append(qlr.pact.ReplaceConvLinearPACTRule(filter_convs, **conv_kwargs))
    rhos.append(qlr.pact.ReplaceActPACTRule(filter_htanh, signed=True, **signed_act_kwargs))
    rhos.append(qlr.pact.ReplaceActPACTRule(filter_relu, signed=False, **unsigned_act_kwargs))

    lwg = qlw.LightweightGraph(net)
    lwe = qlw.LightweightEditor(lwg)

    lwe.startup()
    for rho in rhos:
        lwe.set_lwr(rho)
        lwe.apply()
    lwe.shutdown()

    return lwg.net


def get_pact_controllers(net : nn.Module, schedules : dict, kwargs_linear : dict = {}, kwargs_activation : dict = {}):
    filter_conv = TypeFilter(PACTConv2d) | TypeFilter(PACTConv1d)
    filter_act = TypeFilter(PACTAsymmetricAct) | TypeFilter(PACTUnsignedAct)
    net_nodes = LightweightGraph.build_nodes_list(net)
    conv_modules = [n.module for n in filter_conv(net_nodes)]
    act_modules = [n.module for n in filter_act(net_nodes)]

    print("act_modules")
    print(act_modules)

    lin_ctrl = PACTLinearController(conv_modules, schedules["linear"], **kwargs_linear)
    act_ctrl = PACTActController(act_modules, schedules["activation"], **kwargs_activation)

    return lin_ctrl, act_ctrl
