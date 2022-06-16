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

import torch
import torch.nn as nn

from typing import Union, Optional, List, Tuple, Type


class DownsampleBranch(nn.Module):

    def __init__(self,
                 in_planes:  int,
                 out_planes: int,
                 stride:     int):

        super(DownsampleBranch, self).__init__()
        # pointwise convolution (possible spatial downsampling)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), stride=(stride, stride), padding=0, bias=False)
        self.bn   = nn.BatchNorm2d(out_planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return x


class BasicBlock(nn.Module):

    expansion_factor: int = 1

    def __init__(self,
                 input_planes:  int,
                 hidden_planes: int,
                 output_planes: int,
                 stride:        int,
                 capacity:      float,
                 n_groups:      int,
                 downsample:    Union[torch.nn.Module, None] = None):

        # validate inputs
        if output_planes != (hidden_planes * BasicBlock.expansion_factor):
            raise ValueError

        if not((capacity == 1.0) and (n_groups == 1)):
            raise ValueError("BasicBlock only supports `n_groups == 1` and `capacity == 1.0`.")

        super(BasicBlock, self).__init__()

        # define downsampling/identity branch
        self.downsample = downsample if downsample is not None else nn.Identity()

        # define residual branch
        hidden_planes = int(hidden_planes * capacity) * n_groups
        # layer that (maybe) performs spatial downsampling
        self.conv1 = nn.Conv2d(input_planes, hidden_planes, kernel_size=(3, 3), stride=(stride, stride), padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(hidden_planes)
        self.relu1 = nn.ReLU(inplace=True)
        # layer that preserves the shape
        self.conv2 = nn.Conv2d(hidden_planes, output_planes, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(hidden_planes)

        # post-merge activation
        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # downsampling/identity branch
        identity = self.downsample(x)

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
        x = self.relu_out(x)

        return x


class BottleneckBlock(nn.Module):

    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion_factor: int = 4

    def __init__(self,
                 input_planes:  int,
                 hidden_planes: int,
                 output_planes: int,
                 stride:        int,  # perform downsampling?
                 capacity:      float,
                 n_groups:      int,
                 downsample:    Optional[torch.nn.Module] = None):

        # validate inputs
        if output_planes != (hidden_planes * BottleneckBlock.expansion_factor):
            raise ValueError

        super(BottleneckBlock, self).__init__()

        # define downsampling/identity branch
        self.downsample = downsample if downsample is not None else nn.Identity()

        # define residual branch
        # structural hyper-parameters
        group_planes  = int(hidden_planes * capacity)
        hidden_planes = group_planes * n_groups
        # point-wise inflating convolution
        self.conv1 = nn.Conv2d(input_planes, hidden_planes, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.bn1   = nn.BatchNorm2d(hidden_planes)
        self.relu1 = nn.ReLU(inplace=True)
        # depthwise-separable convolution (possible spatial downsampling)
        self.conv2 = nn.Conv2d(hidden_planes, hidden_planes, kernel_size=(3, 3), stride=(stride, stride), padding=1, groups=n_groups, bias=False)
        self.bn2   = nn.BatchNorm2d(hidden_planes)
        self.relu2 = nn.ReLU(inplace=True)
        # point-wise deflating convolution
        self.conv3 = nn.Conv2d(hidden_planes, output_planes, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.bn3   = nn.BatchNorm2d(output_planes)

        # post-merge activation
        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # downsampling/identity branch
        identity = self.downsample(x)

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
        x = self.relu_out(x)

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


class ResNet(nn.Module):

    def __init__(self,
                 config:    str,
                 capacity:  float = 1.0,
                 n_groups:  int = 1,
                 n_classes: int = 1000,
                 seed:      int = -1,
                 pretrained: Union[bool, str] = False):

        # validate input
        if config not in _CONFIGS.keys():
            raise ValueError

        super(ResNet, self).__init__()

        # build the network
        pilot_out_channels = 64                                  # How many channels should the `torch.Tensor` exiting `self.pilot` have?
        self.pilot = ResNet.make_pilot(pilot_out_channels)
        block_class = _CONFIGS[config]['block_class']
        block_cfgs  = _CONFIGS[config]['block_cfgs']
        features_out_channels = 512 * block_class.expansion_factor  # How many channels should the `torch.Tensor` exiting `self.features` have?
        self.features   = ResNet.make_features(block_class, block_cfgs, pilot_out_channels, features_out_channels, capacity, n_groups)
        self.avgpool    = ResNet.make_avgpool()
        self.classifier = ResNet.make_classifier(features_out_channels, n_classes)

        self._initialize_weights(seed)

        if isinstance(pretrained, bool):
            if pretrained is True:  # load weights from torchvision model
                self.preload_torchvision(config)
        elif isinstance(pretrained, str):  # should be the name of an existing checkpoint file
            self.load_state_dict(torch.load(pretrained))
        else:
            raise TypeError

    @staticmethod
    def make_pilot(out_channels: int) -> nn.Sequential:

        modules = []

        # standard convolutional layer
        modules += [nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=(7, 7), stride=(2, 2), padding=1, bias=False)]
        modules += [nn.BatchNorm2d(out_channels)]
        modules += [nn.ReLU(inplace=True)]
        # max pooling
        modules += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

        return nn.Sequential(*modules)

    @staticmethod
    def make_features(block_class:            Union[Type[BasicBlock], Type[BottleneckBlock]],
                      block_cfgs:             List[Tuple[int, int, int]],
                      features_input_planes:  int,
                      features_output_planes: int,
                      capacity:               float,
                      n_groups:               int) -> nn.Sequential:

        def make_block_sequence(n_blocks:         int,
                                block_class:      Union[Type[BasicBlock], Type[BottleneckBlock]],
                                fb_input_planes:  int,  # first block's input planes
                                fb_hidden_planes: int,
                                fb_output_planes: int,  # non-expanded first block's output planes
                                fb_stride:        int,  # first block's stride
                                capacity:         float,
                                n_groups:         int) -> Tuple[nn.Sequential, int]:

            blocks = []

            # build the first block in the sequence
            # solve structural hyper-parameters
            input_planes  = fb_input_planes
            hidden_planes = fb_hidden_planes
            output_planes = fb_output_planes * block_class.expansion_factor
            stride        = fb_stride
            # build block
            downsample = None if ((stride == 1) and (input_planes == output_planes)) else DownsampleBranch(input_planes, output_planes, stride)
            blocks += [block_class(input_planes, hidden_planes, output_planes, stride=stride, capacity=capacity, n_groups=n_groups, downsample=downsample)]

            # build the remaining blocks
            # solve structural hyper-parameters (they are the same for all the blocks)
            input_planes  = output_planes
            hidden_planes = output_planes // block_class.expansion_factor
            output_planes = output_planes
            stride = 1
            for _ in range(1, n_blocks):
                downsample = None  # remaining blocks always use identity branches
                blocks += [block_class(input_planes, hidden_planes, output_planes, stride=stride, capacity=capacity, n_groups=n_groups, downsample=downsample)]

            return nn.Sequential(*blocks), output_planes

        # assemble the collection of block sequences
        block_sequences = []

        input_planes = features_input_planes
        for (n_blocks, n_planes, stride) in block_cfgs:
            hidden_planes = n_planes
            output_planes = n_planes
            block_seq, output_planes = make_block_sequence(n_blocks, block_class, input_planes, hidden_planes, output_planes, stride, capacity=capacity, n_groups=n_groups)
            block_sequences += [block_seq]
            input_planes = output_planes

        assert input_planes == features_output_planes

        return nn.Sequential(*block_sequences)

    @staticmethod
    def make_avgpool() -> nn.AdaptiveAvgPool2d:
        return nn.AdaptiveAvgPool2d((1, 1))

    @staticmethod
    def make_classifier(in_channels: int,
                        n_classes:   int) -> nn.Linear:
        in_features = in_channels * 1 * 1
        return nn.Linear(in_features=in_features, out_features=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pilot(x)
        x = self.features(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x

    def _initialize_weights(self, seed: int = -1):

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

    def preload_torchvision(self, config: str):

        # utility functions to ease loading of PyTorch's pretrained models
        def resnet_replace_key(k: str) -> str:

            if k.startswith('conv1'):
                return k.replace('conv1', 'pilot.0')
            elif k.startswith('bn1'):
                return k.replace('bn1', 'pilot.1')
            else:
                pass

            for n in range(1, 5):
                k = k.replace(f'layer{n}', f'features.{n - 1}')
            k = k.replace('downsample.0', 'downsample.conv')
            k = k.replace('downsample.1', 'downsample.bn')
            k = k.replace('fc', 'classifier')

            return k

        def dict_keys_match(d1, d2):
            match = True
            match &= all(k in d2.keys() for k in d1.keys())
            match &= all(k in d1.keys() for k in d2.keys())
            return match

        def convert_torchvision_to_ql_resnet_state_dict(state_dict: dict):
            ql_state_dict = {resnet_replace_key(k): v for k, v in state_dict.items()}
            return ql_state_dict

        import torchvision.models as tvm
        import pathlib

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
