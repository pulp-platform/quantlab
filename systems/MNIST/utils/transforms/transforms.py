# 
# transforms.py
# 
# Author(s):
# Philip Wiese <wiesep@student.ethz.ch>
# 
# Copyright (c) 2023 ETH Zurich.
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

from typing import Union, Optional

import torch
from torch import nn
from torchvision.transforms import Normalize

from torchvision.transforms import Compose
from torchvision.transforms import RandomHorizontalFlip  # statistical augmentation transforms
from torchvision.transforms import Lambda
from torchvision.transforms import Pad
from torchvision.transforms import CenterCrop
from torchvision.transforms import RandomCrop            # "evil" transforms combining statistical augmentation with structural aspects
from torchvision.transforms import ToTensor              # structural transforms

from quantlib.algorithms.pact import PACTAsymmetricAct
from quantlib.algorithms.pact.util import almost_symm_quant

MNISTSTATS =\
    {
        'normalize':
            {
                'mean': (0.1307, ),
                'std':  (0.3081, )
            },
        'quantize':
            {
                'min': -0.4240739,
                'max': 2.8215435,
                'eps': 0.025356385856866837
            }
    }


class MNISTNormalize(Normalize):
    def __init__(self):
        super(MNISTNormalize, self).__init__(**MNISTSTATS['normalize'])


class MNISTNormalizeHomogeneous(Normalize):
    def __init__(self):
        mean = torch.mean(torch.Tensor(MNISTSTATS['normalize']['mean']))
        std  = torch.mean(torch.Tensor(MNISTSTATS['normalize']['std']))
        super(MNISTNormalizeHomogeneous, self).__init__(mean=mean, std=std)

class Transform(Compose):
    r"""MNIST normalizing transform with optional augmentation. Uses per-channel
    normalization parameters.
    """
    def __init__(self, augment: bool, crop_size : int = 32, padding : int = 8):

        transforms = []
        if augment:
            transforms.append(RandomHorizontalFlip())
            transforms.append(RandomCrop(crop_size, padding=padding))
        else:
            transforms.append(Pad(padding=padding))
            transforms.append(CenterCrop(crop_size))

        transforms.append(ToTensor())
        transforms.append(MNISTNormalize())

        super(Transform, self).__init__(transforms)

class MNISTPACTQuantTransform(Compose):

    """Extend a CIFAR-10 transform to quantize its outputs.

    The input can be fake-quantized (`quantize == 'fake'`) or true-quantized
    (`quantize == 'int'`).
    """
    def __init__(self, augment: bool, crop_size : int = 32, padding : int = 8, quantize='none', n_q=256, pad_channels : Optional[int] = None, clip : bool = False):

        transforms = []
        transforms.append(Transform(augment, crop_size=crop_size, padding=padding))
        if quantize in ['fake', 'int']:
            transforms.append(PACTAsymmetricAct(n_levels=n_q, symm=True, learn_clip=False, init_clip='max', act_kind='identity'))
            quantizer = transforms[-1]
            # set clip_lo to negative max abs of MNIST
            maximum_abs = torch.max(torch.tensor([v for v in MNISTSTATS['quantize'].values()]).abs())
            clip_lo, clip_hi = almost_symm_quant(maximum_abs, n_q)
            quantizer.clip_lo.data = clip_lo
            quantizer.clip_hi.data = clip_hi
            quantizer.started |= True
        if quantize == 'int':
            eps = transforms[-1].get_eps()
            div_by_eps = lambda x: x/eps
            transforms.append(Lambda(div_by_eps))
        if pad_channels is not None and pad_channels != 3:
            assert pad_channels > 1, "Can't pad MNIST data to <1 channels!"
            pad_img = lambda x: nn.functional.pad(x, (0,0,0,0,0,pad_channels-1), mode='constant', value=0.)
            transforms.append(Lambda(pad_img))
        if clip:
            do_clip = lambda x: nn.functional.relu(x)
            transforms.append(Lambda(do_clip))

        super(MNISTPACTQuantTransform, self).__init__(transforms)
