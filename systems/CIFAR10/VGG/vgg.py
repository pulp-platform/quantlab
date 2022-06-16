# 
# vgg.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
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


_CONFIGS = {
    'VGG8':  ['M', 256, 256, 'M', 512, 512, 'M'],
    'VGG9':  [128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'VGG11': [128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):

    def __init__(self,
                 config:    str,
                 capacity:  float = 1.0,
                 use_bn:    bool = False,
                 n_classes: int = 10,
                 seed:      int = -1,
                 pretrained: str = None) -> None:

        # validate inputs
        config = config.upper()  # canonicalise
        if config not in _CONFIGS.keys():
            raise ValueError

        if capacity <= 0.0:
            raise ValueError  # must be positive

        super(VGG, self).__init__()

        # build the network
        self.pilot      = VGG.make_pilot(config, capacity, use_bn)
        self.features   = VGG.make_features(config, capacity, use_bn)
        self.avgpool    = VGG.make_avgpool(config)
        self.classifier = VGG.make_classifier(config, capacity, use_bn, n_classes)

        self._initialize_weights(seed=seed)

        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained))

    @staticmethod
    def make_pilot(config:   str,
                   capacity: float,
                   use_bn:   bool) -> nn.Sequential:

        in_channels = 3
        out_channels = 64 if config == 'VGG19' else 128
        out_channels = int(out_channels * capacity)

        modules = []

        modules += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=not use_bn)]
        modules += [nn.BatchNorm2d(out_channels)] if use_bn else []
        modules += [nn.ReLU(inplace=True)]

        return nn.Sequential(*modules)

    @staticmethod
    def make_features(config:          str,
                      capacity:        float,
                      use_bn_features: bool) -> nn.Sequential:

        in_channels = 64
        in_channels = in_channels if config == 'VGG19' else (in_channels * 2)
        in_channels = int(in_channels * capacity)

        modules = []

        for v in _CONFIGS[config]:

            if v == 'M':
                modules += [nn.MaxPool2d(kernel_size=2, stride=2)]  # in VGG networks, spatial down-sampling is achieved via max pooling

            else:
                out_channels = int(v * capacity)
                modules += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=not use_bn_features)]
                modules += [nn.BatchNorm2d(out_channels)] if use_bn_features else []
                modules += [nn.ReLU(inplace=True)]
                in_channels = out_channels

        return nn.Sequential(*modules)

    @staticmethod
    def make_avgpool(config: str):
        return nn.AdaptiveAvgPool2d((1, 1)) if config == 'VGG19' else nn.AdaptiveAvgPool2d((4, 4))

    @staticmethod
    def make_classifier(config:    str,
                        capacity:  float,
                        use_bn:    bool,
                        n_classes: int) -> nn.Sequential:

        in_channels = int(512 * capacity)

        modules = []

        if config == 'VGG19':
            in_features = in_channels * 1 * 1
            modules += [nn.Linear(in_features=in_features, out_features=n_classes)]

        else:
            in_features = in_channels * 4 * 4
            # first linear
            modules += [nn.Linear(in_features=in_features, out_features=1024, bias=not use_bn)]
            modules += [nn.BatchNorm1d(1024)] if use_bn else []
            modules += [nn.ReLU(inplace=True)]
            modules += [] if use_bn else [nn.Dropout()]
            # second linear
            modules += [nn.Linear(in_features=1024, out_features=1024, bias=not use_bn)]
            modules += [nn.BatchNorm1d(1024)] if use_bn else []
            modules += [nn.ReLU(inplace=True)]
            modules += [] if use_bn else [nn.Dropout()]
            # last linear (the "real" classifier)
            modules += [nn.Linear(in_features=1024, out_features=n_classes)]

        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pilot(x)
        x = self.features(x)
        x = self.avgpool(x)

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
