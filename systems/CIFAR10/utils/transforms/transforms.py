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
from torch import nn
from torchvision.transforms import Normalize

from torchvision.transforms import Compose
from torchvision.transforms import RandomHorizontalFlip  # statistical
# augmentation transforms
from torchvision.transforms import Lambda
from torchvision.transforms import RandomCrop            # "evil" transforms combining statistical augmentation with structural aspects
from torchvision.transforms import ToTensor              # structural transforms


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
    r"""CIFAR10 normalizing transform with optional augmentation. Uses per-channel
    normalization parameters.
    """
    def __init__(self, augment: bool, crop_size : int = 32, padding : int = 4):

        transforms = []
        if augment:
            transforms.append(RandomCrop(crop_size, padding=padding))
            transforms.append(RandomHorizontalFlip())

        transforms.append(ToTensor())
        transforms.append(CIFAR10Normalize())

        super(TransformA, self).__init__(transforms)

class TransformB(Compose):
    r"""CIFAR10 normalizing transform with optional augmentation. Uses homogeneous
    normalization parameters instead of per-channel parameters.
    """
    def __init__(self, augment: bool, crop_size : int = 32, padding : int = 4):

        transforms = []
        if augment:
            transforms.append(RandomCrop(crop_size, padding=padding))
            transforms.append(RandomHorizontalFlip())

        transforms.append(ToTensor())
        transforms.append(CIFAR10NormalizeHomogeneous())

        super(TransformB, self).__init__(transforms)

class CIFAR10PACTQuantTransform(Compose):

    """Extend a CIFAR-10 transform to quantize its outputs.

    The input can be fake-quantized (`quantize == 'fake'`) or true-quantized
    (`quantize == 'int'`).
    """
    def __init__(self, augment: bool, crop_size : int = 32, padding : int = 4, quantize='none', n_q=256, pad_channels : Optional[int] = None, clip : bool = False):

        transforms = []
        transforms.append(TransformA(augment, crop_size=crop_size, padding=padding))
        if quantize in ['fake', 'int']:
            transforms.append(PACTAsymmetricAct(n_levels=n_q, symm=True, learn_clip=False, init_clip='max', act_kind='identity'))
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
        if pad_channels is not None and pad_channels != 3:
            assert pad_channels > 3, "Can't pad CIFAR10 data to <3 channels!"
            pad_img = lambda x: nn.functional.pad(x, (0,0,0,0,0,pad_channels-3), mode='constant', value=0.)
            transforms.append(Lambda(pad_img))
        if clip:
            do_clip = lambda x: nn.functional.relu(x)
            transforms.append(Lambda(do_clip))

        super(CIFAR10PACTQuantTransform, self).__init__(transforms)
