#
# tokenizer.py
#
# Author(s):
# Philip Wiese <wiesep@student.ethz.ch>
#
# Copyright (c) 2023 ETH Zurich.
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

# Source: https://github.com/lucidrains/vit-pytorch/blob/500e23105a294b55a585462deab1884af264888a/vit_pytorch/cct.py#L58

import torch
from torch import nn


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class Tokenizer(nn.Module):

    def __init__(self,
                 kernel_size,
                 stride,
                 padding,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 n_conv_layers=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 in_planes=64,
                 activation=None,
                 max_pool=True,
                 conv_bias=False):
        super().__init__()

        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        n_filter_list_pairs = zip(n_filter_list[:-1], n_filter_list[1:])

        self.conv_layers = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(chan_in,
                          chan_out,
                          kernel_size=(kernel_size, kernel_size),
                          stride=(stride, stride),
                          padding=(padding, padding),
                          bias=conv_bias), nn.BatchNorm2d(chan_out),
                nn.Identity() if not exists(activation) else activation(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride, padding=pooling_padding
                             ) if max_pool else nn.Identity()) for chan_in, chan_out in n_filter_list_pairs
        ])

        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x: torch.Tensor):
        x = self.conv_layers(x)

        x = x.transpose(3, 1)
        b, _, _, c = x.shape
        x = x.reshape(b, -1, c)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)