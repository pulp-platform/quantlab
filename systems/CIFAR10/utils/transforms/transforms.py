# 
# transforms.py
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
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import RandomCrop, RandomHorizontalFlip
from torchvision.transforms import ToTensor


CIFAR10STATS =\
    {
        'normalize':
            {
                'mean': (0.4914, 0.4822, 0.4465),
                'std':  (0.2470, 0.2430, 0.2610)
            },
        'quantize':
            {
                'min': -1.989473,
                'max': 2.130864,
                'eps': 0.016647374257445335
            }
    }


class CIFAR10Normalize(Normalize):
    def __init__(self):
        super(CIFAR10Normalize, self).__init__(**CIFAR10STATS['normalize'])


class CIFAR10NormalizeHomogeneous(Normalize):
    def __init__(self):
        mean = torch.mean(torch.Tensor(CIFAR10STATS['normalize']['mean']))
        std  = torch.mean(torch.Tensor(CIFAR10STATS['normalize']['std']))
        super(CIFAR10NormalizeHomogeneous, self).__init__(mean=mean, std=std)


class TransformA(Compose):

    def __init__(self, augment: bool):

        transforms = []
        if augment:
            transforms.append(RandomCrop(32, padding=4))
            transforms.append(RandomHorizontalFlip())

        transforms.append(ToTensor())
        transforms.append(CIFAR10Normalize())

        super(TransformA, self).__init__(transforms)
