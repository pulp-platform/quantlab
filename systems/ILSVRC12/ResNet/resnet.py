# 
# resnet.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# Georg Rutishauser <georgr@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2021 ETH Zurich.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

import pathlib
import os

import torch
import torch.nn as nn

import torchvision.models as tvm

from typing import Union, List, Tuple


class DownsampleBranch(nn.Module):

    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 stride: int):

        super(DownsampleBranch, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn   = nn.BatchNorm2d(out_planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv(x)
        x = self.bn(x)

        return x


class BasicBlock(nn.Module):

    expansion_factor: int = 1

    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 stride: int,
                 n_groups: int,
                 group_capacity: int = 1,
                 downsample: Union[torch.nn.Module, None] = None):

        super(BasicBlock, self).__init__()

        if n_groups > 1 or group_capacity > 1:
            raise ValueError("``BasicBlock`` only supports ``n_groups == 1`` and ``group_capacity == 1``.")

        self.downsample = downsample  # if ``None``, this will be the identity function (i.e., a skip branch)

        group_planes  = out_planes * group_capacity
        hidden_planes = n_groups * group_planes

        self.conv1 = nn.Conv2d(in_planes, hidden_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(hidden_planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(hidden_planes, hidden_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(hidden_planes)

        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # downsampling/skip branch
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        # residual branch
        # layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # layer 2
        x = self.conv2(x)
        x = self.bn2(x)

        # merge branches
        x += identity
        x = self.relu2(x)

        return x


class BottleneckBlock(nn.Module):

    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion_factor: int = 4

    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 stride: int,
                 n_groups: int,
                 group_capacity: int = 1,
                 downsample: Union[torch.nn.Module, None] = None):

        super(BottleneckBlock, self).__init__()

        self.downsample = downsample  # if ``None``, this will be the identity function (i.e., a skip branch)

        group_planes  = out_planes * group_capacity
        hidden_planes = n_groups * group_planes

        self.conv1 = nn.Conv2d(in_planes, hidden_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1   = nn.BatchNorm2d(hidden_planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(hidden_planes, hidden_planes, kernel_size=3, stride=stride, padding=1, groups=n_groups, bias=False)
        self.bn2   = nn.BatchNorm2d(hidden_planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(hidden_planes, out_planes * self.expansion_factor, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3   = nn.BatchNorm2d(out_planes * self.expansion_factor)

        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # downsampling/skip branch
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        # residual branch
        # layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # layer 3
        x = self.conv3(x)
        x = self.bn3(x)

        # merge branches
        x += identity
        x = self.relu3(x)

        return x


_CONFIGS = {
    'ResNet18':  {'block_class': BasicBlock,
                  'block_cfgs': [( 2,  64, 1),
                                 ( 2, 128, 2),
                                 ( 2, 256, 2),
                                 ( 2, 512, 2)]},
    'ResNet26':  {'block_class': BottleneckBlock,
                  'block_cfgs': [( 2,  64, 1),
                                 ( 2, 128, 2),
                                 ( 2, 256, 2),
                                 ( 2, 512, 2)]},
    'ResNet34':  {'block_class': BasicBlock,
                  'block_cfgs': [( 3,  64, 1),
                                 ( 4, 128, 2),
                                 ( 6, 256, 2),
                                 ( 3, 512, 2)]},
    'ResNet50':  {'block_class': BottleneckBlock,
                  'block_cfgs': [( 3,  64, 1),
                                 ( 4, 128, 2),
                                 ( 6, 256, 2),
                                 ( 3, 512, 2)]},
    'ResNet101': {'block_class': BottleneckBlock,
                  'block_cfgs': [( 3,  64, 1),
                                 ( 4, 128, 2),
                                 (23, 256, 2),
                                 ( 3, 512, 2)]},
    'ResNet152': {'block_class': BottleneckBlock,
                  'block_cfgs': [( 3,  64, 1),
                                 ( 8, 128, 2),
                                 (36, 256, 2),
                                 ( 3, 512, 2)]},
}

def resnet_replace_key(k):
    if k.startswith('conv1'):
        return k.replace('conv1', 'pilot.0')
    if k.startswith('bn1'):
        return k.replace('bn1', 'pilot.1')
    for n in range(1, 5):
        k = k.replace(f'layer{n}', f'features.{n-1}')
    k = k.replace('downsample.0', 'downsample.conv')
    k = k.replace('downsample.1', 'downsample.bn')
    k = k.replace('fc', 'classifier')
    return k

def dict_keys_match(d1, d2):
    match = True
    match &= all(k in d2.keys() for k in d1.keys())
    match &= all(k in d1.keys() for k in d2.keys())
    return match

def convert_torchvision_to_ql_resnet_state_dict(state_dict : dict):
    ql_state_dict = {resnet_replace_key(k):v for k,v in state_dict.items()}
    return ql_state_dict

class ResNet(nn.Module):

    def __init__(self,
                 config: str,
                 n_groups: int = 1,
                 group_capacity: int = 1,
                 num_classes: int = 1000,
                 seed : int = -1,
                 pretrained : Union[bool, str] = False):

        super(ResNet, self).__init__()

        block_class = _CONFIGS[config]['block_class']
        block_cfgs  = _CONFIGS[config]['block_cfgs']

        out_channels_pilot    = 64
        in_planes_features    = out_channels_pilot
        out_planes_features   = 512 * block_class.expansion_factor
        out_channels_features = out_planes_features

        self.pilot      = self._make_pilot(out_channels_pilot)
        self.maxpool    = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.features   = self._make_features(block_cfgs, block_class, in_planes_features, out_planes_features, n_groups, group_capacity)
        self.avgpool    = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(out_channels_features, num_classes)
        if pretrained == True: # load weights from torchvision model
            self.preload_torchvision(config)
        elif pretrained: # if `pretrained` is string, load from checkpoint
            self.load_state_dict(torch.load(pretrained))
        else: # randomly initialize weights
            self._initialize_weights(seed)

    def _make_pilot(self, out_channels_pilot: int) -> nn.Sequential:

        modules  = []
        modules += [nn.Conv2d(3, out_channels_pilot, kernel_size=7, stride=2, padding=3, bias=False)]
        modules += [nn.BatchNorm2d(out_channels_pilot)]
        modules += [nn.ReLU(inplace=True)]

        return nn.Sequential(*modules)

    def _make_features(self,
                       block_cfgs: List[Tuple[int, int, int]],
                       block: Union[BasicBlock, BottleneckBlock],
                       in_planes_features: int,
                       out_planes_features: int,
                       n_groups: int,
                       group_capacity: int) -> nn.Sequential:

        def _make_block_seq(n_blocks: int,
                            block: Union[BasicBlock, BottleneckBlock],
                            in_planes: int,
                            out_planes: int,
                            stride: int,
                            n_groups: int,
                            group_capacity: int) -> Tuple[nn.Sequential, int]:

            blocks = []

            # build first block in the sequence (possibly use non-identity skip branch)
            exp_out_planes = out_planes * block.expansion_factor
            if stride != 1 or in_planes != exp_out_planes:
                downsample = DownsampleBranch(in_planes, exp_out_planes, stride)
            else:
                downsample = None

            blocks += [block(in_planes, out_planes, stride=stride, n_groups=n_groups, group_capacity=group_capacity, downsample=downsample)]
            in_planes = exp_out_planes

            # build remaining blocks (they always use skip branches)
            for _ in range(1, n_blocks):
                blocks += [block(in_planes, out_planes, stride=1, n_groups=n_groups, group_capacity=group_capacity, downsample=None)]

            return nn.Sequential(*blocks), exp_out_planes

        block_seqs = []
        in_planes = in_planes_features
        for block_cfg in block_cfgs:
            block_seq, exp_out_planes = _make_block_seq(block_cfg[0], block, in_planes, out_planes=block_cfg[1], stride=block_cfg[2], n_groups=n_groups, group_capacity=group_capacity)
            block_seqs += [block_seq]
            in_planes = exp_out_planes

        assert in_planes == out_planes_features

        return nn.Sequential(*block_seqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pilot(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x

    def _initialize_weights(self, seed : int = -1):

        if seed >= 0:
            torch.manual_seed(seed)

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch, so that each
        # residual block behaves like an identity. This improves the accuracy
        # of the model by ~0.2/0.3%, according to https://arxiv.org/abs/1706.02677.
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)
            elif isinstance(m, BottleneckBlock):
                nn.init.constant_(m.bn3.weight, 0)

    def preload_torchvision(self, config : str):
        assert config in _CONFIGS and config != 'ResNet26', f"ResNet: Unsupported preload config: {config}"
        pretrained_path = pathlib.Path(__file__).parent.joinpath('pretrained')
        pretrained_path.mkdir(exist_ok=True)
        ckpt_path = pretrained_path.joinpath(f"{config.lower()}_torchvision.ckpt")
        if not ckpt_path.exists():
            tv_model_fn = getattr(tvm, config.lower())
            tv_model = tv_model_fn(pretrained=True)
            ql_state_dict = convert_torchvision_to_ql_resnet_state_dict(tv_model.state_dict())
            assert dict_keys_match(self.state_dict(), ql_state_dict), f"Error loading pretrained ResNet state_dict for config '{config}'!"
            self.load_state_dict(ql_state_dict)
            torch.save(self.state_dict(), str(ckpt_path))
        else:
            self.load_state_dict(torch.load(str(ckpt_path)))
