import torch
import torch.nn as nn


class DSCNN(torch.nn.Module):

    def __init__(self, use_bn_features: bool = True, use_bias_classifier: bool = False, num_classes: int = 12, seed: int = -1):

        super(DSCNN, self).__init__()

        self.pilot = self._make_pilot(64, use_bn_features)
        self.features = self._make_features(64, use_bn_features)
        self.avgpool = nn.AvgPool2d(kernel_size=(25, 5), stride=1)
        self.classifier = nn.Linear(64, num_classes, bias=use_bias_classifier)  # logits

        self._initialize_weights(seed=seed)

    @staticmethod
    def _make_pilot(out_channels: int, use_bn: bool) -> nn.Sequential:

        pad = nn.ConstantPad2d((1, 1, 5, 5), value=0.0)
        modules = []
        modules += [nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(10, 4), stride=(2, 2), bias=not use_bn)]
        modules += [nn.BatchNorm2d(64)]
        modules += [nn.ReLU(inplace=True)]

        return nn.Sequential(*[pad, nn.Sequential(*modules)])

    @staticmethod
    def _make_dw_layer(n_channels: int, use_bn: bool) -> nn.Sequential:

        modules = []
        modules += [nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=(3, 3), stride=(1, 1), groups=64, bias=not use_bn)]
        modules += [nn.BatchNorm2d(n_channels)] if use_bn else []
        modules += [nn.ReLU(inplace=True)]

        return nn.Sequential(*modules)

    @staticmethod
    def _make_pw_layer(n_channels: int, use_bn: bool) -> nn.Sequential:

        modules = []
        modules += [nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=(1, 1), stride=(1, 1), bias=not use_bn)]
        modules += [nn.BatchNorm2d(n_channels)] if use_bn else []
        modules += [nn.ReLU(inplace=True)]

        return nn.Sequential(*modules)

    @staticmethod
    def _make_block(n_channels: int, use_bn: bool) -> nn.Sequential:

        modules = []
        modules += [nn.ConstantPad2d((1, 1, 1, 1), value=0.)]
        modules += [DSCNN._make_dw_layer(n_channels, use_bn)]
        modules += [DSCNN._make_pw_layer(n_channels, use_bn)]

        return nn.Sequential(*modules)

    @staticmethod
    def _make_features(n_channels: int, use_bn: bool) -> nn.Sequential:

        modules = []
        modules += [DSCNN._make_block(n_channels, use_bn)]
        modules += [DSCNN._make_block(n_channels, use_bn)]
        modules += [DSCNN._make_block(n_channels, use_bn)]
        modules += [DSCNN._make_block(n_channels, use_bn)]

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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)