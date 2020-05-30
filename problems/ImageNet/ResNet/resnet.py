import os
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# model_urls = {
#     'se_resnet26': 'https://data.vision.ee.ethz.ch/kmaninis/share/MTL//models/se_resnet26-5eb336d20.pth',
#     'se_resnet50': 'https://data.vision.ee.ethz.ch/kmaninis/share/MTL//models/se_resnet50-ad8889f9f.pth',
#     'se_resnet101': 'https://data.vision.ee.ethz.ch/kmaninis/share/MTL//models/se_resnet101-8dbb64f8e.pth',
# }
# TODO: Move this to config

__all__ = ['ResNet', 'resnet26', 'resnet50', 'resnet101']


models_dir = './models'

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        expansion = 4
        self.stride = stride

        # channel compression
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        # channel decompression
        self.conv3 = nn.Conv2d(planes, planes * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.downsample = downsample
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu3(out)

        return out


class ResNet(nn.Module):

    def __init__(self, layers, num_classes=1000, socket3x3=False):
        self.inplanes = 64
        block = Bottleneck
        super(ResNet, self).__init__()

        if not socket3x3:
            adapter = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        else:
            adapter = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        self.socket = adapter

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion: # planes * block.expansion = outplanes; in Bottleneck, block.expansion seems hard-coded to 4
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.socket(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def get_state_dict(model_name, remote=False):
    # find and load checkpoint
    if remote:
        raise NotImplementedError
    else:
        checkpoint = torch.load(os.path.join(models_dir, model_name + '.pth'), map_location=lambda storage, loc: storage)
    checkpoint = checkpoint['state_dict']

    # handle DataParallel
    if 'module.' in list(checkpoint.keys())[0]:
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k.replace('module.', '')  # remove `module.`
            new_state_dict[name] = v
    else:
        new_state_dict = checkpoint

    return new_state_dict


def resnet26(pretrained=False, **kwargs):
    """Construct a ResNet-26 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        depthwise (bool): If True, create a model with depthwise separable convolutions.
    """
    model = ResNet(Bottleneck, [2, 2, 2, 2], **kwargs)

    if pretrained:
        print('Loading ResNet-26 ImageNet...')
        model_name = 'resnet26'
        state_dict = get_state_dict(model_name)
        # Load pre-trained IN model
        model.load_state_dict(state_dict)
        print('\b done!')

    return model


def resnet50(pretrained=False, **kwargs):
    """Construct a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        depthwise (bool): If True, create a model with depthwise separable convolutions.
    """

    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:
        print('Loading ResNet-50 ImageNet...')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        print('\b done!')

    return model


def resnet101(pretrained=False, **kwargs):
    """Construct a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        depthwise (bool): If True, create a model with depthwise separable convolutions.
    """

    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

    if pretrained:
        print('Loading ResNet-101 ImageNet...')
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
        print('\b done!')

    return model
