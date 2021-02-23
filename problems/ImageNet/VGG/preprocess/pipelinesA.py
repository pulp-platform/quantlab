from torchvision.transforms import Resize, CenterCrop
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip
from torchvision.transforms import ToTensor, Normalize
from torchvision.transforms import Compose

from .transforms import _ImageNet, ColorJitter, Lighting


def get_pipelines(augment):

    valid_pipe = Compose([Resize(256),
                       CenterCrop(224),
                       ToTensor(),
                       Normalize(**_ImageNet['Normalize'])])

    # since
    if not augment:
        train_pipe = valid_pipe
    else:
        train_pipe = Compose([RandomResizedCrop(224),
                              RandomHorizontalFlip(),
                              ToTensor(),
                              ColorJitter(),
                              Lighting(_ImageNet['PCA']),
                              Normalize(**_ImageNet['Normalize'])])

    pipelines = {
        'training':   train_pipe,
        'validation': valid_pipe
    }

    return pipelines
