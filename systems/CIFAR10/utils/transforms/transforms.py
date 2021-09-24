# 
# transforms.py
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

from torchvision.transforms import Normalize
from torchvision.transforms import Compose
from torchvision.transforms import RandomHorizontalFlip  # statistical augmentation transforms
from torchvision.transforms import RandomCrop            # "evil" transforms combining statistical augmentation with structural aspects
from torchvision.transforms import ToTensor              # structural transforms

from quantlib.algorithms.pact import PACTAsymmetricAct
from quantlib.algorithms.pact.util import almost_symm_quant



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


class CIFAR10AugmentTransform(Compose):

    def __init__(self, augment: bool):

        transforms = []
        if augment:
            transforms.append(RandomCrop(32, padding=4))
            transforms.append(RandomHorizontalFlip())

        transforms.append(ToTensor())
        transforms.append(CIFAR10Normalize())

        super(CIFAR10AugmentTransform, self).__init__(transforms)



# an extended AugmentTransform which can also quantize the normalized input.
# input is fake-quantized for quantize='fake'
# input is integerized for quantize='int'
class CIFAR10PACTQuantTransform(Compose):
    def __init__(self, augment : bool, quantize='none', n_q=256):
        transforms = []
        transforms.append(CIFAR10AugmentTransform(augment))
        if quantize in ['fake', 'int']:
            transforms.append(PACTAsymmetricAct(n_levels=n_q, symm=True, learn_clip=False))
            quantizer = transforms[-1]
            # set clip_lo to negative max abs of CIFAR10
            maximum_abs = torch.max(torch.tensor([v for v in CIFAR10STATS['quantize'].values()]).abs())
            clip_lo, clip_hi = almost_symm_quant(maximum_abs, n_q)
            quantizer.clip_lo.data = clip_lo
            quantizer.clip_hi.data = clip_hi
            quantizer.started |= True
        if quantize == 'int':
            eps = transforms[-1].get_eps()
            div_by_eps = lambda x: x/eps
            transforms.append(Lambda(div_by_eps))

        super(CIFAR10AugmentTransform, self).__init__(transforms)
