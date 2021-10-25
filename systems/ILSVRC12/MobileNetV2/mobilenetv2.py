# 
# mobilenetv2.py
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

from typing import Union


_CONFIGS = {
    'standard': [
        # t,  c, n, s
        [1,  16, 1, 1],
        [6,  24, 2, 2],
        [6,  32, 3, 2],
        [6,  64, 4, 2],
        [6,  96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1]
    ],
}


class Conv2dBNActivation(nn.Sequential):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 activation: Union[torch.nn.Module, None] = None) -> None:

        modules = []
        modules += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, groups=groups, bias=False)]
        modules += [nn.BatchNorm2d(out_channels)]
        modules += [nn.ReLU6(inplace=True) if activation is None else activation(inplace=True)]

        super().__init__(*modules)


class Conv2dBN(nn.Sequential):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1) -> None:

        modules = []
        modules += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, groups=groups, bias=False)]
        modules += [nn.BatchNorm2d(out_channels)]

        super().__init__(*modules)


class InvertedResidual(nn.Module):

    def __init__(self,
                 input_planes: int,
                 output_planes: int,
                 stride: int,
                 expansion_factor: int) -> None:

        assert stride in {1, 2}
        super(InvertedResidual, self).__init__()

        # build residual branch
        layers = []
        hidden_planes = input_planes * expansion_factor
        if expansion_factor > 1:
            # point-wise convolution
            layers += [Conv2dBNActivation(input_planes, hidden_planes, kernel_size=1)]
        # depth-wise convolution
        layers += [Conv2dBNActivation(hidden_planes, hidden_planes, kernel_size=3, stride=stride, groups=hidden_planes)]
        # point-wise convolution
        layers += [Conv2dBN(hidden_planes, output_planes, kernel_size=1)]

        self.residual_branch = nn.Sequential(*layers)

        # should I use skip (i.e., identity) branch?
        self._use_skip_branch = (stride == 1) and (input_planes == output_planes)

    def forward(self, x):

        if self._use_skip_branch:
            return x + self.residual_branch(x)
        else:  # this is a bottleneck layer (i.e., skip-branch-free)
            return self.residual_branch(x)


class MobileNetV2(nn.Module):

    def __init__(self, config: str, capacity: float = 1.0, round_to_closest_multiple_of: int = 8, num_classes: int = 1000, pretrained : str = None, seed : int = -1):

        super(MobileNetV2, self).__init__()

        out_channels_pilot    = MobileNetV2._make_divisible_by(32 * capacity, divisor=round_to_closest_multiple_of)
        in_planes_features    = out_channels_pilot
        out_planes_features   = MobileNetV2._make_divisible_by(1280 * max(1.0, capacity), divisor=round_to_closest_multiple_of)
        out_channels_features = out_planes_features

        self.pilot      = self._make_pilot(out_channels_pilot)
        self.features   = self._make_features(config, capacity, round_to_closest_multiple_of, in_planes_features, out_planes_features)
        self.avgpool    = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = self._make_classifier(out_channels_features, num_classes)

        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained))
        else:
            self._initialize_weights(seed)

    @staticmethod
    def _make_divisible_by(dilated_num_channels: float, divisor: int, min_num_channels: Union[int, None] = None) -> int:

        if min_num_channels is None:
            min_num_channels = divisor
        assert min_num_channels % divisor == 0

        num_channels = max(min_num_channels, (int(dilated_num_channels + divisor / 2) // divisor) * divisor)

        if (dilated_num_channels - num_channels) / dilated_num_channels >= 0.1:
            num_channels += divisor

        return num_channels

    def _make_pilot(self, out_channels_pilot: int) -> Conv2dBNActivation:
        return Conv2dBNActivation(3, out_channels_pilot, kernel_size=3, stride=2)

    def _make_features(self,
                       config: str,
                       capacity: float,
                       round_to_closest_multiple_of: int,
                       in_planes_features: int,
                       out_planes_features: int) -> nn.Sequential:

        blocks = []
        in_planes = in_planes_features
        for t, c, n, s in _CONFIGS[config]:
            out_planes = MobileNetV2._make_divisible_by(c * capacity, divisor=round_to_closest_multiple_of)
            for block_id in range(0, n):
                stride = s if block_id == 0 else 1
                blocks += [InvertedResidual(in_planes, out_planes, stride, t)]
                in_planes = out_planes

        blocks += [Conv2dBNActivation(in_planes, out_planes_features, kernel_size=1)]

        return nn.Sequential(*blocks)

    def _make_classifier(self, out_channels_features: int, num_classes: int) -> nn.Sequential:

        modules = []
        modules += [nn.Dropout(0.2)]
        modules += [nn.Linear(out_channels_features, num_classes)]

        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pilot(x)
        x = self.features(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x

    def _initialize_weights(self, seed : int = -1):

        if seed >= 0:
            torch.manual_seed(seed)

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

