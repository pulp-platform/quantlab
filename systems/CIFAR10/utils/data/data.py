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

import torch
import torchvision

from systems.utils.data import default_dataset_cv_split

def load_data_set(partition: str,
                  path_data: str,
                  n_folds: int,
                  current_fold_id: int,
                  cv_seed: int,
                  transform: torchvision.transforms.Compose) -> torch.utils.data.Dataset:

    if partition in {'train', 'valid'}:

        if n_folds > 1:  # this is a cross-validation experiment

            dataset = torchvision.datasets.CIFAR10(root=path_data, train=True, download=True, transform=transform)
            train_fold_indices, valid_fold_indices = default_dataset_cv_split(dataset=dataset, n_folds=n_folds, current_fold_id=current_fold_id, cv_seed=cv_seed)

            if partition == 'train':
                dataset = torch.utils.data.Subset(dataset, train_fold_indices)
            elif partition == 'valid':
                dataset = torch.utils.data.Subset(dataset, valid_fold_indices)

        else:
            dataset = torchvision.datasets.CIFAR10(root=path_data, train=True if partition == 'train' else False, download=True, transform=transform)

    else:
        assert partition == 'test'
        raise NotImplementedError

    return dataset

