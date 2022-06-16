# 
# alexnet.py
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


class AlexNet(nn.Module):

    def __init__(self,
                 use_bn:    bool,
                 n_classes: int = 1000,
                 seed:      int = -1) -> None:

        super(AlexNet, self).__init__()

        # build the network
        self.features   = AlexNet.make_features(use_bn)
        self.avgpool    = AlexNet.make_avgpool()
        self.classifier = AlexNet.make_classifier(use_bn, n_classes)

        self._initialize_weights(seed)

    @staticmethod
    def make_features(use_bn: bool) -> nn.Sequential:

        modules = []

        # conv 1
        modules += [nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=2, bias=not use_bn)]
        modules += [nn.BatchNorm2d(64)] if use_bn else []
        modules += [nn.ReLU(inplace=True)]
        # max pool
        modules += [nn.MaxPool2d(kernel_size=3, stride=2)]

        # conv 2
        modules += [nn.Conv2d(64, 192, kernel_size=(5, 5), padding=2, bias=not use_bn)]
        modules += [nn.BatchNorm2d(192)] if use_bn else []
        modules += [nn.ReLU(inplace=True)]
        # max pool
        modules += [nn.MaxPool2d(kernel_size=3, stride=2)]

        # conv 3
        modules += [nn.Conv2d(192, 384, kernel_size=(3, 3), padding=1, bias=not use_bn)]
        modules += [nn.BatchNorm2d(384)] if use_bn else []
        modules += [nn.ReLU(inplace=True)]

        # conv 4
        modules += [nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1, bias=not use_bn)]
        modules += [nn.BatchNorm2d(256)] if use_bn else []
        modules += [nn.ReLU(inplace=True)]

        # conv 5
        modules += [nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1, bias=not use_bn)]
        modules += [nn.BatchNorm2d(256)] if use_bn else []
        modules += [nn.ReLU(inplace=True)]
        # max pool
        modules += [nn.MaxPool2d(kernel_size=3, stride=2)]

        return nn.Sequential(*modules)

    @staticmethod
    def make_avgpool() -> nn.AdaptiveAvgPool2d:
        return nn.AdaptiveAvgPool2d((6, 6))

    @staticmethod
    def make_classifier(use_bn:    bool,
                        n_classes: int) -> nn.Sequential:

        in_features = 256 * 6 * 6

        modules = []

        # first linear
        modules += [] if use_bn else [nn.Dropout()]
        modules += [nn.Linear(in_features=in_features, out_features=4096, bias=not use_bn)]
        modules += [nn.BatchNorm1d(4096)] if use_bn else []
        modules += [nn.ReLU(inplace=True)]

        # second linear
        modules += [] if use_bn else [nn.Dropout()]
        modules += [nn.Linear(in_features=4096, out_features=4096, bias=not use_bn)]
        modules += [nn.BatchNorm1d(4096)] if use_bn else []
        modules += [nn.ReLU(inplace=True)]

        # last linear (the "real" classifier)
        modules += [nn.Linear(in_features=4096, out_features=n_classes)]

        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

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
