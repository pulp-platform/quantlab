# 
# alexnet.py
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


class AlexNet(nn.Module):

    def __init__(self, use_bn: bool, num_classes: int = 1000) -> None:

        super(AlexNet, self).__init__()

        self.features   = self._make_features(use_bn)
        self.avgpool    = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = self._make_classifier(num_classes)

    def _make_features(self, use_bn: bool) -> nn.Sequential:

        modules = []

        # conv 1
        modules += [nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=not use_bn)]
        modules += [nn.BatchNorm2d(64)] if use_bn else []
        modules += [nn.ReLU(inplace=True)]
        # max pool
        modules += [nn.MaxPool2d(kernel_size=3, stride=2)]
        # conv 2
        modules += [nn.Conv2d(64, 192, kernel_size=5, padding=2, bias=not use_bn)]
        modules += [nn.BatchNorm2d(192)] if use_bn else []
        modules += [nn.ReLU(inplace=True)]
        # max pool
        modules += [nn.MaxPool2d(kernel_size=3, stride=2)]
        # conv 3
        modules += [nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=not use_bn)]
        modules += [nn.BatchNorm2d(384)] if use_bn else []
        modules += [nn.ReLU(inplace=True)]
        # conv 4
        modules += [nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=not use_bn)]
        modules += [nn.BatchNorm2d(256)] if use_bn else []
        modules += [nn.ReLU(inplace=True)]
        # conv 5
        modules += [nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=not use_bn)]
        modules += [nn.BatchNorm2d(256)] if use_bn else []
        modules += [nn.ReLU(inplace=True)]
        # max pool
        modules += [nn.MaxPool2d(kernel_size=3, stride=2)]

        return nn.Sequential(*modules)

    def _make_classifier(self, num_classes: int) -> nn.Sequential:

        modules = []

        # dropout
        modules += [nn.Dropout()]
        # linear 1
        modules += [nn.Linear(256 * 6 * 6, 4096)]
        modules += [nn.ReLU(inplace=True)]
        # dropout
        modules += [nn.Dropout()]
        # linear 2
        modules += [nn.Linear(4096, 4096)]
        modules += [nn.ReLU(inplace=True)]
        # linear 3
        modules += [nn.Linear(4096, num_classes)]

        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.features(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x

