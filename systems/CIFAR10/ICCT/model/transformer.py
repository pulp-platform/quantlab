#
# transformers.py
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

# Source:
#  1 - https://iis-git.ee.ethz.ch/victor.jung/transformers_chb_mit/-/blob/quant/pytorch_code/transformer_pytorch.py
#  2 - https://iis-git.ee.ethz.ch/pulptransformers/tinytransformers/-/blob/master/tinytransformer_layers.py

import torch
from torch import nn


class LambdaLayer(torch.nn.Module):

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class Transpose(torch.nn.Module):

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def forward(self, x):
        return torch.transpose(x, self.x, self.y)


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout = 0., bias = True, activationFunction: nn.Module = nn.GELU):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias = bias),
            activationFunction(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias = bias),
        )

        bound1 = 1 / (dim**.5)
        bound2 = 1 / (hidden_dim**.5)
        nn.init.uniform_(self.ff[0].weight, -bound1, bound1)
        nn.init.uniform_(self.ff[0].bias, -bound1, bound1)
        nn.init.uniform_(self.ff[3].weight, -bound2, bound2)
        nn.init.uniform_(self.ff[0].bias, -bound2, bound2)

    def forward(self, x):
        return self.ff(x)


class VanillaAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.Softmax = nn.Softmax(dim = -1)

    def forward(self, q, k, v, dim, mask = None, dropout = None):

        # scores = self.Softmax(q @ k.transpose(-2, -1))
        scores = self.Softmax(torch.matmul(q, k.transpose(-2, -1)))

        if dropout is not None:
            scores = dropout(scores)

        return torch.matmul(scores, v)


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 dim,
                 heads,
                 head_dim = None,
                 AttentionMechanism = VanillaAttention,
                 bias = True,
                 share_kv = False,
                 fix_q = False,
                 *args,
                 **kwargs):
        super().__init__()

        if head_dim is None:
            self.dim = int(dim // heads)
        else:
            self.dim = head_dim
        self.h = heads
        self.inner_dim = int(heads * self.dim)
        self.out_dim = dim

        if fix_q:
            self.WQ = nn.Identity
        else:
            self.WQ = nn.Linear(dim, self.inner_dim, bias = bias)

        self.WK = nn.Linear(dim, self.inner_dim, bias = bias)

        if share_kv:
            self.WV = self.WK
        else:
            self.WV = nn.Linear(dim, self.inner_dim, bias = bias)

        self.attention = AttentionMechanism(*args, **kwargs)
        self.out = nn.Linear(self.inner_dim, self.out_dim)

    def forward(self, q, k, v):
        mask = None

        b, s, dh = k.shape

        q = self.WQ(q)
        q = q.view(b, -1, self.h, self.dim)
        q = torch.transpose(q, 1, 2)

        k = self.WK(k)
        k = k.view(b, -1, self.h, self.dim)
        k = torch.transpose(k, 1, 2)

        v = self.WV(v)
        v = v.view(b, -1, self.h, self.dim)
        v = torch.transpose(v, 1, 2)

        scores = self.attention(q, k, v, self.dim, mask, None)
        concat = scores.transpose(1, 2).contiguous().view(scores.shape[0], -1, self.inner_dim)

        return self.out(concat)


class Attention(nn.Module):

    def __init__(self, dim, heads = 8, dim_head = 64, bias=True):
        super().__init__()

        # Make sure to use the same names as in quantlib/editing/fx/passes/pact/harmonize.py:736
        self.out_dim = dim
        self.dim = dim_head
        self.h = heads

        self.mhsa = MultiHeadAttention(dim, heads, dim_head, VanillaAttention, bias=bias)

    def forward(self, x):
        return self.mhsa(x, x, x)


class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., attention_bias = True, mlp_bias = True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, bias = attention_bias)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout, bias = mlp_bias))
                ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
