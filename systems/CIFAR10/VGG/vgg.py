import torch
import torch.nn as nn


_CONFIGS = {
    'VGG8': ['M', 256, 256, 'M', 512, 512, 'M'],
    'VGG9': [128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'VGG11': [128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'],
}


class VGG(nn.Module):

    def __init__(self, config: str, capacity: int = 1, use_bn: bool = False, num_classes: int = 10, seed: int = -1) -> None:

        super(VGG, self).__init__()

        self.pilot      = self._make_pilot(capacity, use_bn)
        self.features   = self._make_features(config, capacity, use_bn)
        self.avgpool    = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = self._make_classifier(capacity, num_classes)

        self._initialize_weights(seed=seed)

    @staticmethod
    def _make_pilot(capacity: int, use_bn: bool) -> nn.Sequential:

        modules = []
        modules += [nn.Conv2d(3, 128 * capacity, kernel_size=3, padding=1, bias=not use_bn)]
        modules += [nn.BatchNorm2d(128 * capacity)] if use_bn else []
        modules += [nn.ReLU(inplace=True)]

        return nn.Sequential(*modules)

    @staticmethod
    def _make_features(config: str, capacity: int, use_bn: bool) -> nn.Sequential:

        modules = []
        in_channels = 128 * capacity
        for v in _CONFIGS[config]:
            if v == 'M':
                modules += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                out_channels = v * capacity
                modules += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn)]
                modules += [nn.BatchNorm2d(out_channels)] if use_bn else []
                modules += [nn.ReLU(inplace=True)]
                in_channels = out_channels

        return nn.Sequential(*modules)

    @staticmethod
    def _make_classifier(capacity: int, num_classes: int) -> nn.Sequential:

        modules = []
        modules += [nn.Linear(512 * capacity * 4 * 4, 1024)]
        modules += [nn.ReLU(inplace=True)]
        modules += [nn.Dropout()]
        modules += [nn.Linear(1024, 1024)]
        modules += [nn.ReLU(inplace=True)]
        modules += [nn.Dropout()]
        modules += [nn.Linear(1024, num_classes)]

        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pilot(x)
        x = self.features(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)  # https://stackoverflow.com/questions/57234095/what-is-the-difference-of-flatten-and-view-1-in-pytorch

        x = self.classifier(x)

        return x

    def _initialize_weights(self, seed: int):

        if seed >= 0:
            torch.manual_seed(seed)

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)