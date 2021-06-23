# 
# vgg_pact.py
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
from torch import nn


__all__ = ['VGG7']


class VGG7(nn.Module):

    def __init__(self, capacity : int, seed : int):
        super(VGG7, self).__init__()
        self.features = self._make_features(capacity)
        n_fc = 1024
        c3 = capacity * 128 * 4
        self.classifier = nn.Sequential(
            nn.Linear(c3*4*4, n_fc),
            nn.ReLU(inplace=True),
            nn.Linear(n_fc, n_fc),
            nn.ReLU(inplace=True),
            nn.Linear(n_fc, 10)
        )
        self._initialize_weights(seed)

    @staticmethod
    def _make_features(capacity : int):
        c0 = 3
        c1 = int(capacity * 128)
        c2 = int(capacity * 128 * 2)
        c3 = int(capacity * 128 * 4)

        in_ch = c0
        layers = []
        for c in (c1, c2, c3):
            layers.append(nn.Conv2d(in_ch, c, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(c))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(c))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = c

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
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
                nn.init.constant_(m.bias, 0)

