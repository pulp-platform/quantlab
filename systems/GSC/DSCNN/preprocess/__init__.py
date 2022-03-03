import os
import torch
import torchvision

from .dataset import GSCDataset
from systems.utils.data.cvsplit import default_dataset_cv_split


__all__ = [
    'load_data_set',
    'NullTransform',
]


def load_data_set(partition: str,
                  path_data: os.PathLike,
                  n_folds: int,
                  current_fold_id: int,
                  cv_seed: int,
                  transform: torchvision.transforms.Compose) -> torch.utils.data.Dataset:

    if partition in {'train', 'valid'}:

        if n_folds > 1:

            dataset = GSCDataset(path_data=path_data, partition=partition)
            train_fold_indices, valid_fold_indices = default_dataset_cv_split(dataset=dataset, n_folds=n_folds, current_fold_id=current_fold_id, cv_seed=cv_seed)

            if partition == 'train':
                dataset = torch.utils.data.Subset(dataset, train_fold_indices)
            elif partition == 'valid':
                dataset = torch.utils.data.Subset(dataset, valid_fold_indices)

        else:
            dataset = GSCDataset(path_data=path_data, partition=partition)

    else:
        assert partition == 'test'
        dataset = GSCDataset(path_data=path_data, partition=partition)

    return dataset


# TODO: since QuantLab always expects a transform, as long as the pre-processing will be hard-coded into the `Dataset` itself we need to pass a proxy
class NullTransform(torchvision.transforms.Compose):

    def __init__(self):
        super().__init__([])
