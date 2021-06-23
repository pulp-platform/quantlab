# 
# __init__.py
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

import os
import torch
import torchvision

from .transform_a import get_transform_a


__all__ = [
    'dataset_load',
    'get_transform_a',
]


def dataset_load(path_data: str, transform: torchvision.transforms.Compose, train: bool = True) -> torch.utils.data.Dataset:

    path_dataset = os.path.join(os.path.realpath(path_data), 'train' if train else 'val')
    dataset = torchvision.datasets.ImageFolder(path_dataset, transform)

    return dataset

