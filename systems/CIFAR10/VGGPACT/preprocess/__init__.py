import os

import torch
import torchvision

from .transform_a import TransformA
from systems.utils.data import default_dataset_cv_split


__all__ = [
    'load_data_set',
    'TransformA',
]


def load_data_set(partition: str,
                  path_data: str,
                  n_folds: int,
                  current_fold_id: int,
                  cv_seed: int,
                  transform: torchvision.transforms.Compose) -> torch.utils.data.Dataset:

    if partition in {'train', 'valid'}:
        realpath_data = os.path.realpath(path_data)
        if n_folds > 1:  # this is a cross-validation experiment
            dataset = torchvision.datasets.CIFAR10(root=realpath_data, train=True, download=True, transform=transform)
            train_fold_indices, valid_fold_indices = default_dataset_cv_split(dataset=dataset, n_folds=n_folds, current_fold_id=current_fold_id, cv_seed=cv_seed)

            if partition == 'train':
                dataset = torch.utils.data.Subset(dataset, train_fold_indices)
            elif partition == 'valid':
                dataset = torch.utils.data.Subset(dataset, valid_fold_indices)

        else:
            dataset = torchvision.datasets.CIFAR10(root=realpath_data, train=(partition == 'train'), download=True, transform=transform)

    else:
        assert partition == 'test'
        raise NotImplementedError

    return dataset
