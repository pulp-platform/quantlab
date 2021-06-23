# 
# ana.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2021 ETH Zurich. All rights reserved.
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


def all_ana_recipe(net:            nn.Module,
                   quantizer_spec: dict,
                   noise_type:     str) -> nn.Module:

    # define filters
    # type-based filters
    filter_conv2d = qlw.rules.filters.TypeFilter(nn.Conv2d)
    filter_linear = qlw.rules.filters.TypeFilter(nn.Linear)
    filter_relu   = qlw.rules.filters.TypeFilter(nn.ReLU)
    # name-based filters
    filter_pilot      = qlw.rules.filters.NameFilter('pilot')
    filter_features   = qlw.rules.filters.NameFilter('features')
    filter_classifier = qlw.rules.filters.NameFilter('classifier')
    filter_last_node  = qlw.rules.filters.NameFilter('classifier.6')

    # define rules
    # 2D convolutions
    filter_conv2d_pilot_or_features = filter_conv2d & (filter_pilot | filter_features)
    rho_conv2d = qlw.rules.ana.ReplaceConv2dANAConv2dRule(filter_=filter_conv2d_pilot_or_features, quantizer_spec=quantizer_spec, noise_type=noise_type)
    # linear maps
    filter_linear_classifier_no_last = filter_linear & filter_classifier & (-filter_last_node)
    rho_linear = qlw.rules.ana.ReplaceLinearANALinearRule(filter_=filter_linear_classifier_no_last, quantizer_spec=quantizer_spec, noise_type=noise_type)
    # ReLUs
    filter_relu_pilot_or_features_or_classifier = filter_relu & (filter_pilot | filter_features | filter_classifier)
    rho_relu = qlw.rules.ana.ReplaceReLUANAActivationRule(filter_relu_pilot_or_features_or_classifier, quantizer_spec=quantizer_spec, noise_type=noise_type)

    # edit
    lwgraph  = qlw.LightweightGraph(net)
    lweditor = qlw.LightweightEditor(lwgraph)

    lweditor.startup()
    lweditor.set_lwr(rho_conv2d)
    lweditor.apply()
    lweditor.set_lwr(rho_linear)
    lweditor.apply()
    lweditor.set_lwr(rho_relu)
    lweditor.apply()
    lweditor.shutdown()

    return lwgraph.net


def all_ana_controller(net:       nn.Module,
                       ctrl_spec: list) -> List[qa.Controller]:
    anactrl = qa.ana.ANAController(net, ctrl_spec)
    return [anactrl]

