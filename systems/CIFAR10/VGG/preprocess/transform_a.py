# 
# transform_a.py
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

from torchvision.transforms import Compose
from torchvision.transforms import RandomHorizontalFlip  # statistical augmentation transforms
from torchvision.transforms import RandomCrop            # "evil" transforms combining statistical augmentation with structural aspects
from torchvision.transforms import ToTensor              # structural transforms

from systems.CIFAR10.utils.transforms import CIFAR10Normalize  # public CIFAR-10 transforms


class TransformA(Compose):

    def __init__(self, augment: bool):

        transforms = []
        if augment:
            transforms.append(RandomCrop(32, padding=4))
            transforms.append(RandomHorizontalFlip())

        transforms.append(ToTensor())
        transforms.append(CIFAR10Normalize())

        super(TransformA, self).__init__(transforms)
