# 
# simplecnn.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
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

import torch
import torch.nn as nn


_CONFIGS = {
    'standard':  ['M', 32, 'M', 64, 'M', 128, 'M'],
}


class simpleCNN(nn.Module):

    def __init__(self, config: str, capacity: int = 1, use_bn_features: bool = False, use_bn_classifier: bool = False, pretrained : str = None, num_classes: int = 10, seed: int = -1) -> None:

        super(simpleCNN, self).__init__()

        self.pilot      = self._make_pilot(config, capacity, use_bn_features)
        self.features   = self._make_features(config, capacity, use_bn_features)
        self.classifier = self._make_classifier(config, capacity, use_bn_classifier, num_classes)
        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained), strict=False)
        else:
            self._initialize_weights(seed=seed)

    @staticmethod
    def _make_pilot(config: str, capacity: int, use_bn_features: bool) -> nn.Sequential:

        out_channels = 16
        modules = []
        modules += [nn.Conv2d(1, out_channels * capacity, kernel_size=3, padding=1, bias=not use_bn_features)]
        modules += [nn.BatchNorm2d(out_channels * capacity)] if use_bn_features else []
        modules += [nn.ReLU(inplace=True)]

        return nn.Sequential(*modules)

    @staticmethod
    def _make_features(config: str, capacity: int, use_bn_features: bool) -> nn.Sequential:

        in_channels  = 16
        in_channels *= capacity

        modules = []
        for v in _CONFIGS[config]:
            if v == 'M':
                modules += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                out_channels = v * capacity
                modules += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn_features)]
                modules += [nn.BatchNorm2d(out_channels)] if use_bn_features else []
                modules += [nn.ReLU(inplace=True)]
                in_channels = out_channels

        return nn.Sequential(*modules)


    @staticmethod
    def _make_classifier(config: str, capacity: int, use_bn_classifier: bool, num_classes: int) -> nn.Sequential:

        modules = []
        
        modules += [] if use_bn_classifier else [nn.Dropout()]
        modules += [nn.Linear(512 * capacity, num_classes, bias=not use_bn_classifier)]
        modules += [nn.BatchNorm1d(num_classes)] if use_bn_classifier else []
        modules += [nn.LogSoftmax(1)]

        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pilot(x)
        x = self.features(x)

        x = x.view(x.size(0), -1)  # https://stackoverflow.com/questions/57234095/what-is-the-difference-of-flatten-and-view-1-in-pytorch

        x = self.classifier(x)

        return x

    def _initialize_weights(self, seed: int):

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
