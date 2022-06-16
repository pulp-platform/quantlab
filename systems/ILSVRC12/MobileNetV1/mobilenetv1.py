#
# mobilenetv1.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# Georg Rutishauser <georgr@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2022 ETH Zurich.
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

import torch
import torch.nn as nn

from typing import Union, Tuple


_CONFIGS = {
    'STANDARD': [
        ( 2, 1),
        ( 4, 2),
        ( 4, 1),
        ( 8, 2),
        ( 8, 1),
        (16, 2),
        (16, 1),
        (16, 1),
        (16, 1),
        (16, 1),
        (16, 1),
        (32, 2),
        (32, 1)
    ]
}


class MobileNetV1(nn.Module):

    def __init__(self,
                 config:     str,
                 capacity:   float = 1.0,
                 activation: str = 'ReLU',
                 n_classes:  int = 1000,
                 pretrained: str = None,
                 seed:       int = -1):

        # validate inputs
        config = config.upper()  # canonicalise
        if config not in _CONFIGS.keys():
            raise ValueError

        activation = activation.lower()  # canonicalise
        if activation == 'relu':
            activation_class = nn.ReLU
        elif activation == 'relu6':
            activation_class = nn.ReLU6
        else:
            raise ValueError

        super(MobileNetV1, self).__init__()

        # build the network
        base_width      = int(32 * capacity)
        self.pilot      = MobileNetV1.make_pilot(base_width, activation_class)
        self.features   = MobileNetV1.make_features(config, base_width, activation_class)
        self.avgpool    = MobileNetV1.make_avgpool()
        self.classifier = MobileNetV1.make_classifier(config, base_width, n_classes)

        self._initialize_weights(seed)

        if pretrained:
            self.load_state_dict(torch.load(pretrained))

    @staticmethod
    def make_standard_convolution_layer(in_channels:      int,
                                        out_channels:     int,
                                        stride:           Union[int, Tuple[int, ...]],
                                        activation_class: type) -> nn.Sequential:

        modules = []

        modules += [nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)]
        modules += [nn.BatchNorm2d(out_channels)]
        modules += [activation_class(inplace=True)]

        return nn.Sequential(*modules)

    @staticmethod
    def make_depthwise_separable_convolution_block(in_channels:      int,
                                                   out_channels:     int,
                                                   stride:           Union[int, Tuple[int, ...]],
                                                   activation_class: type) -> nn.Sequential:

        modules = []

        # depthwise
        modules += [nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=in_channels, bias=False)]
        modules += [nn.BatchNorm2d(in_channels)]
        modules += [activation_class(inplace=True)]

        # pointwise
        modules += [nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)]
        modules += [nn.BatchNorm2d(out_channels)]
        modules += [activation_class(inplace=True)]

        return nn.Sequential(*modules)

    @staticmethod
    def make_pilot(base_width:       int,
                   activation_class: type) -> nn.Sequential:

        in_channels = 3
        out_channels = base_width

        return MobileNetV1.make_standard_convolution_layer(in_channels=in_channels,
                                                           out_channels=out_channels,
                                                           stride=2,  # we start with a spatial down-sampling
                                                           activation_class=activation_class)

    @staticmethod
    def make_features(config:           str,
                      base_width:       int,
                      activation_class: type) -> nn.Sequential:

        modules = []

        in_channels = base_width
        for n_channels_multiplier, stride in _CONFIGS[config]:
            out_channels = base_width * n_channels_multiplier
            modules += [MobileNetV1.make_depthwise_separable_convolution_block(in_channels=in_channels,
                                                                               out_channels=out_channels,
                                                                               stride=stride,
                                                                               activation_class=activation_class)]
            in_channels = out_channels

        return nn.Sequential(*modules)

    @staticmethod
    def make_avgpool() -> nn.AdaptiveAvgPool2d:
        return nn.AdaptiveAvgPool2d((1, 1))

    @staticmethod
    def make_classifier(config:     str,
                        base_width: int,
                        n_classes:  int) -> nn.Linear:

        last_n_channels_multiplier = _CONFIGS[config][-1][0]
        in_channels = last_n_channels_multiplier * base_width
        in_features = in_channels * 1 * 1

        return nn.Linear(in_features=in_features, out_features=n_classes)

    def forward(self, x):

        x = self.pilot(x)
        x = self.features(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x

    def _initialize_weights(self, seed: int = -1):

        if seed >= 0:
            torch.manual_seed(seed)

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
