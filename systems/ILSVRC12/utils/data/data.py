# 
# data.py
# 
# Author(s):
# Georg Rutishauser <georgr@iis.ee.ethz.ch>
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

import os
import torch
import torchvision


def load_ilsvrc12(partition: str,
                  path_data: str,
                  n_folds: int,
                  current_fold_id: int,
                  cv_seed: int,
                  transform: torchvision.transforms.Compose) -> torch.utils.data.Dataset:

    # basic dataset creation function for ILSVRC12
    if n_folds != 1:
        print("Warning: ImageNet 'load_data_set' function does not support cross-validation yet!")
    path_dataset = os.path.join(os.path.realpath(path_data), 'train' if partition == 'train' else 'val')
    dataset = torchvision.datasets.ImageFolder(path_dataset, transform)

    return dataset
