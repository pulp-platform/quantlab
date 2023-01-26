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

# Source: https://iis-git.ee.ethz.ch/victor.jung/transformers_chb_mit/-/blob/quant/pytorch_code/transformer_pytorch.py

import math
import torch
import torch.nn.functional as F
from torch import nn


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):

        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

        bound1 = 1 / (dim ** .5)
        bound2 = 1 / (hidden_dim ** .5)
        nn.init.uniform_(self.net[0].weight, -bound1, bound1)
        nn.init.uniform_(self.net[0].bias, -bound1, bound1)
        nn.init.uniform_(self.net[3].weight, -bound2, bound2)
        nn.init.uniform_(self.net[0].bias, -bound2, bound2)

    def forward(self, x):
        return self.net(x)

class VanillaAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.Softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, dim, mask=None, dropout=None):

        scores = torch.matmul(q, k.transpose(-2, -1)) #/  math.sqrt(dim)

        scores = self.Softmax(scores / math.sqrt(dim))

        if dropout is not None:
            scores = dropout(scores)

        output = torch.matmul(scores, v)

        return output


class MultiHeadAttention(nn.Module):

    def __init__(self, dim, heads, head_dim = None, AttentionMechanism = VanillaAttention, dropout = 0.1, share_kv = False, fix_q = False, *args, **kwargs):
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
            self.WQ = nn.Linear(dim, self.inner_dim, bias=True)

        self.WK = nn.Linear(dim, self.inner_dim, bias=True)

        if share_kv:
            self.WV = self.WK
        else:
            self.WV = nn.Linear(dim, self.inner_dim, bias=True)

        self.AttentionMechanism = AttentionMechanism(*args, **kwargs)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.inner_dim, self.out_dim)

    def forward(self, q, k, v):
        mask = None

        b, s, dh = k.shape

        q = self.WQ(q)

        q = q.view(b, -1, self.h, self.dim)
        q = torch.transpose(q, 1,2)

        k = self.WK(k)
        k = k.view(b, -1, self.h, self.dim)
        k = torch.transpose(k, 1,2)

        v = self.WV(v)
        v = v.view(b, -1, self.h, self.dim)
        v = torch.transpose(v, 1,2)

        scores = self.AttentionMechanism(q, k, v, self.dim, mask, None)

        concat = scores.transpose(1,2).contiguous().view(scores.shape[0], -1, self.inner_dim)

        output = self.out(concat)

        return output

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attention = MultiHeadAttention(dim, heads, dim_head, VanillaAttention, dropout)


    def forward(self, x):
        out = self.attention(x,x,x)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

