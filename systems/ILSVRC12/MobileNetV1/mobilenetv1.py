#
# mobilenetv1.py
# 
# Author(s):
# Georg Rutishauser <georgr@iis.ee.ethz.ch>
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

from typing import Union

# config: (n, stride) where n specifies output channels as n*base_width = n*capacity*32
_CONFIGS = {
    'standard':
    [(2, 1), (4, 2), (4, 1), (8, 2), (8, 1), (16, 2), (16, 1), (16, 1), (16, 1), (16, 1), (16, 1), (32, 2), (32, 1)]
}

class MobileNetV1(nn.Module):
    def __init__(self, capacity : float = 1., config : str = 'standard', n_classes : int = 1000, seed : int = -1, pretrained : str = None, act_fn : str = 'relu'):
        assert config in _CONFIGS.keys(), f"Unknown config {config}"
        assert act_fn in ['relu', 'relu6'], f"Unknown activation function {act_fn}!"
        act = nn.ReLU if act_fn == 'relu' else nn.ReLU6
        super(MobileNetV1, self).__init__()
        base_width = int(capacity*32)
        self.pilot = self.bn_conv(3, base_width, 2, act)

        features = []
        in_ch = base_width
        for n, stride in _CONFIGS[config]:
            features.append(self.dws_conv(in_ch, n*base_width, stride, act))
            in_ch = n*base_width

        self.last_out_ch = in_ch

        features.append(nn.AdaptiveAvgPool2d((1,1)))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Linear(in_ch, n_classes)
        if pretrained:
            self.load_state_dict(torch.load(pretrained))
        else:
            self._initialize_weights(seed)


    @staticmethod
    def bn_conv(in_ch : int, out_ch : int, stride : tuple, act : type):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (3,3), stride=stride, padding=(1,1), bias=False),
            nn.BatchNorm2d(out_ch),
            act(inplace=True)
        )

    @staticmethod
    def dws_conv(in_ch : int, out_ch : int, stride : tuple, act : type):
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, (3,3), stride=stride, padding=(1,1), groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            act(inplace=True),
            nn.Conv2d(in_ch, out_ch, (1,1), stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(out_ch),
            act(inplace=True)
        )

    def forward(self, x):
        x = self.pilot(x)
        x = self.features(x)
        x = x.view(-1, self.last_out_ch)
        return self.classifier(x)

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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


