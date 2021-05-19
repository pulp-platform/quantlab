from torchvision.transforms import Normalize


CIFAR10STATS =\
    {
        'normalize':
            {
                'mean': (0.4914, 0.4822, 0.4465),
                'std':  (0.2470, 0.2430, 0.2610)
            }
    }


class CIFAR10Normalize(Normalize):
    def __init__(self):
        super(CIFAR10Normalize, self).__init__(**CIFAR10STATS['normalize'])
