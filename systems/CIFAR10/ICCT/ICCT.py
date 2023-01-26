#
# simplecnn.py
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

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import quantlib.editing.lightweight as qlw

from .model import *


class ICCT(nn.Module):

    def __init__(
        self,
        embedding_dim=64,
        projection_dim = 64,
        num_heads=4,
        num_layers=1,
        final_pool="mean",
        n_conv_layers=1,
        n_input_channels=3,
        conv_kernel_size=3,
        conv_padding=3,
        conv_stride=2,
        pooling_kernel_size=3,
        pooling_padding=0,
        pooling_stride=2,
        pretrained: str = "",
        seed: int = -1,
        *args,
        **kwargs,
    ) -> None:
        super(ICCT, self).__init__()

        self.pool = final_pool

        if seed >= 0:
            torch.manual_seed(seed)

        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=conv_kernel_size,
                                   stride=conv_stride,
                                   padding=conv_padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=n_conv_layers,
                                   conv_bias=False)

        self.transformer = Transformer(dim=embedding_dim,
                                       depth=num_layers,
                                       heads=num_heads,
                                       dim_head=projection_dim,
                                       mlp_dim=4 * embedding_dim,
                                       dropout=0.0)
                                       
        self.last_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 10),
            nn.LogSoftmax()
        )

        if pretrained != "":
            state_dict = torch.load(pretrained)
            if 'net' in state_dict:
                self.load_state_dict(state_dict['net'])
            else:
                self.load_state_dict(state_dict)
            print(f"[QuantLab] Loading state from {pretrained}")
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x)
        x = self.transformer(x)
        x = self.last_norm(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.classifier(x)

        return x
