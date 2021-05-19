import torch
import torchvision

from .transform_a import get_transform_a


__all__ = [
    'dataset_load',
    'get_transform_a',
]


def dataset_load(path_data: str, transform: torchvision.transforms.Compose, train: bool = True) -> torch.utils.data.Dataset:
    dataset = torchvision.datasets.CIFAR10(root=path_data, train=train, download=True, transform=transform)
    return dataset
