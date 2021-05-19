from torchvision.transforms import Compose
from torchvision.transforms import RandomHorizontalFlip  # statistical augmentation transforms
from torchvision.transforms import RandomResizedCrop  # "evil" transforms combining statistical augmentation with structural aspects
from torchvision.transforms import Resize, CenterCrop, ToTensor  # structural transforms

from systems.ILSVRC12.utils.transforms.transforms import ColorJitter, ILSVRC12Lighting, ILSVRC12Normalize  # public ILSVRC12 transforms


def get_transform_a(train: bool, augment: bool = True) -> Compose:

    if augment and train:
        transform = Compose([RandomResizedCrop(224),
                              RandomHorizontalFlip(),
                              ToTensor(),
                              ColorJitter(),
                              ILSVRC12Lighting(),
                              ILSVRC12Normalize()])
    else:
        transform = Compose([Resize(256),
                             CenterCrop(224),
                             ToTensor(),
                             ILSVRC12Normalize()])

    return transform
