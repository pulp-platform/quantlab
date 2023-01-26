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

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class CCT(nn.Module):
    def __init__(self,
        img_size=32,
        embedding_dim=128,
        n_input_channels=3,
        n_conv_layers=1,
        kernel_size=3,
        stride=2,
        padding=3,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=0,
        pretrained : str = "", seed: int = -1,
        *args, **kwargs,  ) -> None:
        super(CCT, self).__init__(    )

        img_height, img_width = pair(img_size)

        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=n_conv_layers,
                                   conv_bias=False)
        
        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_height,
                                                           width=img_width),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout_rate=0.0,
            attention_dropout=0.0,
            stochastic_depth=0.1,
            num_layers = 1,
            num_heads = 2,
            num_classes=10,
            *args, **kwargs)

       
        if pretrained is not "":
            state_dict = torch.load(pretrained)
            if 'net' in state_dict:
                self.load_state_dict(state_dict['net'])
            else:
                self.load_state_dict(state_dict)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x)
        return self.classifier(x)
    
   