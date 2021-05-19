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
