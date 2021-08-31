# 
# vgg.py
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

import torch
import torch.nn as nn


_CONFIGS = {
    'VGG11': ['M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, config: str, capacity: int = 1, use_bn: bool = False, num_classes: int = 1000, seed : int = -1):

        super(VGG, self).__init__()

        self.pilot      = self._make_pilot(capacity, use_bn)
        self.features   = self._make_features(config, capacity, use_bn)
        self.avgpool    = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = self._make_classifier(capacity, num_classes)

        self._initialize_weights(seed)

    @staticmethod
    def _make_pilot(capacity: int, use_bn: bool) -> nn.Sequential:

        layers = []
        layers += [nn.Conv2d(3, 64 * capacity, kernel_size=3, padding=1, bias=not use_bn)]
        layers += [nn.BatchNorm2d(64 * capacity)] if use_bn else []
        layers += [nn.ReLU(inplace=True)]

        return nn.Sequential(*layers)

    @staticmethod
    def _make_features(config: str, capacity: int, use_bn: bool) -> nn.Sequential:

        layers = []
        in_channels = 64 * capacity
        for v in _CONFIGS[config]:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                out_channels = v * capacity
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn)]
                layers += [nn.BatchNorm2d(out_channels)] if use_bn else []
                layers += [nn.ReLU(inplace=True)]
                in_channels = out_channels

        return nn.Sequential(*layers)

    @staticmethod
    def _make_classifier(capacity: int, num_classes: int) -> nn.Sequential:

        layers = []
        layers += [nn.Linear(512 * capacity * 7 * 7, 4096)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Dropout()]
        layers += [nn.Linear(4096, 4096)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Dropout()]
        layers += [nn.Linear(4096, num_classes)]

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pilot(x)
        x = self.features(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)  # https://stackoverflow.com/questions/57234095/what-is-the-difference-of-flatten-and-view-1-in-pytorch

        x = self.classifier(x)

        return x

    def _initialize_weights(self, seed : int = -1):

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
                nn.init.constant_(m.bias, 0)

