# 
# all_ana.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2022 ETH Zurich. All rights reserved.
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

import torch.nn as nn

import quantlib.algorithms as qa
import quantlib.editing.lightweight as qlw

from typing import List


def all_ana_recipe(net:                 nn.Module,
                   quantizer_spec_conv: dict,
                   quantizer_spec_act:  dict,
                   noise_type:          str,
                   strategy:            str) -> nn.Module:

    # define rules
    # 2D convolutions
    filter_conv2d = qlw.rules.filters.TypeFilter(nn.Conv2d)
    rho_conv2d = qlw.rules.ana.ReplaceConv2dANAConv2dRule(filter_=filter_conv2d, quantizer_spec=quantizer_spec_conv, noise_type=noise_type, strategy=strategy)
    # ReLUs
    filter_relu   = qlw.rules.filters.TypeFilter(nn.ReLU)
    rho_relu = qlw.rules.ana.ReplaceReLUANAActivationRule(filter_relu, quantizer_spec=quantizer_spec_act, noise_type=noise_type, strategy=strategy)

    # edit
    lwgraph  = qlw.LightweightGraph(net)
    lweditor = qlw.LightweightEditor(lwgraph)

    lweditor.startup()
    lweditor.set_lwr(rho_conv2d)
    lweditor.apply()
    lweditor.set_lwr(rho_relu)
    lweditor.apply()
    lweditor.shutdown()

    return lwgraph.net


def all_ana_controller(net:       nn.Module,
                       ctrl_spec: list) -> List[qa.Controller]:
    anactrl = qa.ana.ANAController(net, ctrl_spec)
    return [anactrl]
