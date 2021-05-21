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
