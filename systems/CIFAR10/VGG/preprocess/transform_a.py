from torchvision.transforms import Compose
from torchvision.transforms import RandomHorizontalFlip  # statistical augmentation transforms
from torchvision.transforms import RandomCrop            # "evil" transforms combining statistical augmentation with structural aspects
from torchvision.transforms import ToTensor              # structural transforms

from systems.CIFAR10.utils.transforms import CIFAR10Normalize  # public CIFAR-10 transforms


def get_transform_a(train: bool, augment: bool = True) -> Compose:

    if train and augment:
        transform = Compose([RandomCrop(32, padding=4),
                             RandomHorizontalFlip(),
                             ToTensor(),
                             CIFAR10Normalize()])
    else:
        transform = Compose([ToTensor(),
                             CIFAR10Normalize()])

    return transform
