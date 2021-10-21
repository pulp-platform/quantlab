# 
# precision_search.py
# 
# Author(s):
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

import sys
import os
from typing import Union
from collections import namedtuple
from copy import copy
from functools import reduce
from pathlib import Path

import numpy as np

import torch
from torch import nn
from torch.nn.modules.utils import _pair
from quantlib.editing.lightweight import LightweightGraph, LightweightNode

pool_types = (nn.AvgPool2d, nn.MaxPool2d)
conv_types = (nn.Conv2d, nn.Conv1d)
linear_types = (nn.Linear,)

class QuantConvLayer:
    def __init__(self, node : LightweightNode, in_dims : tuple,  q_in : int = 8, q_out : int = 8, q_wt : int = 8):
        self.node = node
        self.q_in = q_in
        self.q_out = q_out
        self.q_wt = q_wt
        self.in_dims = in_dims

    @property
    def conv(self):
        return self.node.module

    @property
    def out_dims(self):
        # in newer pyTorch versions, we can safely assume that e.g. the 'stride' is
        # an n-tuple in a Convnd
        return tuple(d//s for (d,s) in zip(self.in_dims, self.conv.stride))

    @property
    def in_size(self):
        ic = self.conv.in_channels
        fm_els = ic * np.prod(self.in_dims)
        return int(np.ceil(fm_els * self.q_in/8))

    @property
    def out_size(self):
        oc = self.conv.out_channels
        fm_els = oc * np.prod(self.out_dims)
        return int(np.ceil(fm_els * self.q_out/8))

    @property
    def wt_size(self):
        return int(self.conv.weight.numel() * self.q_wt / 8)


    @property
    def extra_param_size_mem(self):
        # combined size (in bytes) of:
        # - biases (32b)
        # - zero point of input activations (8b)
        # - zero point of output activations (8b)
        # - zero points of weights (16b - why again?)
        # - scale factor M_0 (32b)
        # - shift parameter N_0 for scale factor (8b)

        return 2 + self.conv.out_channels * 11

    @property
    def extra_param_size_st(self):
        # like extra_param_size_mem, but only considers input zero point, as
        # one layer's output is the next layer's input

        return 1 + self.conv.out_channels * 11

    @property
    def tot_param_size(self):
        return self.wt_size + self.extra_param_size_mem

class QuantLinLayer:
    def __init__(self, node : LightweightNode, q_in : int = 8, q_out : int = 8, q_wt : int = 8, split_fac : int = 1):
        self.node = node
        self.q_in = q_in
        self.q_out = q_out
        self.q_wt = q_wt
        self.split_fac = split_fac

    @property
    def lin(self):
        return self.node.module

    @property
    def in_dims(self):
        return (self.lin.in_features,)

    @property
    def out_dims(self):
        return (self.lin.out_features,)

    @property
    def in_size(self):
        return int(np.ceil(self.lin.in_features*self.q_in/8))

    @property
    def out_size(self):
        return int(np.ceil(self.lin.out_features*self.q_out/8))

    @property
    def wt_size(self):
        return int(np.ceil(self.lin.weight.numel()*self.q_wt/8/self.split_fac))

    @property
    def extra_param_size_st(self):
        s = 0
        s += int(np.ceil(4 * self.lin.out_features/self.split_fac)) # bias
        s += int(np.ceil(self.lin.out_features/self.split_fac))     # Z_w - but why only 8b?
        s += 4                         # Z_o or Z_i??? manuele pls eggsblain :DDD
        s += int(np.ceil(2* self.lin.out_features/self.split_fac))  # M_0, N_0??? again pls explain
        return s

    @property
    def extra_param_size_mem(self):
        # just copy _st version, no idea if there should be a difference
        s = 0
        s += 4 * self.lin.out_features # bias
        s += self.lin.out_features     # Z_w - but why only 8b?
        s += 4                         # Z_o or Z_i??? manuele pls eggsblain :DDD
        s += 2* self.lin.out_features  # M_0, N_0??? again pls explain
        return s

    @property
    def tot_param_size(self):
        return self.wt_size + self.extra_param_size_mem



def quant_layers_from_net(net : nn.Module, in_dims : tuple, extra_conv_types=(), extra_pool_types=(), extra_lin_types=(), last_layer_split : int = 1):
    dims = in_dims
    nodes = LightweightGraph.build_nodes_list(net)
    convs = conv_types + extra_conv_types
    pools = pool_types + extra_pool_types
    linears = linear_types + extra_lin_types
    conv_pool_modules = [n for n in nodes if isinstance(n.module, convs + pools + linears)]
    ql = []
    for i, m in enumerate(conv_pool_modules):
        if isinstance(m.module, convs):
            l = QuantConvLayer(m, dims) # default: 8b
            dims = l.out_dims
            ql.append(l)
        elif isinstance(m.module, linears):
            split_fac = 1 if i != len(conv_pool_modules)-1 else last_layer_split
            l = QuantLinLayer(m, split_fac=split_fac)
            dims = l.out_dims
            ql.append(l)
        else:
            dims = tuple(d//s for (d, s) in zip(dims, _pair(m.kernel_size)))

    return ql


def cut_acts(ql_in : list, mem_rw : int, qa_min : int):
    ql = copy(ql_in)
    for l in ql:
        l.q_in = 8
        l.q_out = 8
        l.q_wt = 8

    def cut_bits(l : QuantConvLayer, inp : bool):
        # return true if input/output activation bits should be cut
        if inp:
            q_x1 = l.q_out
            mem_x1 = l.out_size
            q_x2 = l.q_in
            mem_x2 = l.in_size
        else:
            q_x1 = l.q_in
            mem_x1 = l.in_size
            q_x2 = l.q_out
            mem_x2 = l.out_size

        if q_x2 > qa_min:
            return (q_x2 > q_x1) or (q_x2 == q_x1 and mem_x2 >= mem_x1)

        return False

    def layer_fits(l : QuantConvLayer):
        return l.in_size + l.out_size <= mem_rw

    def net_fits():
        return all([layer_fits(l) for l in ql])

    while not net_fits():
        forward_action = False
        backward_action = False
        # forward pass: cut output bits for each layer
        for i, l in enumerate(ql[:-1]):
            while (not layer_fits(l)) and cut_bits(l, False):
                l.q_out = int(l.q_out//2)
                ql[i+1].q_in = l.q_out
                forward_action = True
        # backward pass: cut input bits for each layer
        for i, l in reversed(list(enumerate(ql))):
            if i != 0:
                while (not layer_fits(l)) and cut_bits(l, True):
                    l.q_in = int(l.q_in//2)
                    ql[i-1].q_out = l.q_in
                    backward_action = True
        if (not net_fits()) and not (forward_action or backward_action):
            print("Activation cutting got stuck and will not terminate! Returning original layer list.")
            return ql_in

    print("Activation cutting terminated successfully!")
    return ql


def cut_wts(ql_in : list, mem_ro : int, qw_min : int, delta : float):
    ql = copy(ql_in)

    def net_size():
        params_size = 0
        for l in ql:
            params_size += l.tot_param_size
        return params_size

    def net_fits():
        params_size = net_size()
        return params_size <= mem_ro

    def ratios():
        r = []
        idxs = []
        params_size = net_size()
        for i, l in enumerate(ql):
            if l.q_wt > qw_min:
                r.append(l.wt_size/params_size)
                idxs.append(i)

        return idxs, r

    while not net_fits():
        idxs, rs = ratios()
        r_max = np.max(rs)
        candidate_layers = [i for i, r in zip(idxs, rs) if r > r_max-delta]
        cut_layer = ql[np.min(candidate_layers)]
        cut_layer.q_wt = int(cut_layer.q_wt//2)

    print("Weight cutting terminated successfully!")
    return ql

def cut_acts_wts_orig(ql_in : list, mem_ro : int, mem_rw : int, qw_min : int, qa_min : int, delta : float):
    assert mem_ro, "Please supply a valid 'mem_ro' constraint to cut_acts_wts_orig - you supplied {}".format(mem_ro)
    ql_actcut = cut_acts(ql_in, mem_rw, qa_min)
    return cut_wts(ql_actcut, mem_ro, qw_min, delta)





def cut_acts_wts_pulp(ql_in : list, mem_rw : int, q_min : int, q_min_wts : int = None):

    ql = copy(ql_in)

    def layer_size(idx : int):
        l = ql[idx]
        tot_size = l.in_size + l.out_size + l.tot_param_size
        if idx != len(ql)-1:
            tot_size += ql[idx+1].tot_param_size
        elif isinstance(l, QuantLinLayer) and l.split_fac > 1:
            # if we split a last linear layer, we need to store 2 sets of
            # parameters at a time
            tot_size += l.tot_param_size
        return tot_size

    def layer_fits(idx : int):
        # a layer fits if its inputs, outputs, weights and the weights of the
        # next layer all fit in RAM
        tot_size = layer_size(idx)
        return tot_size <= mem_rw

    def net_fits():
        # the net fits if all the layers fit
        return reduce(lambda a, b: a and b, [layer_fits(i) for i in range(len(ql))])

    def cut_cur(idx : int):
        qnew = int(ql[idx].q_in/2)
        ql[idx].q_in = qnew
        ql[idx-1].q_out = qnew
        ql[idx].q_wt = qnew
        print("Cutting layer {} inputs and weights to {} bits!".format(idx, qnew))

    def cut_next(idx : int):
        qnew = int(ql[idx].q_out/2)
        ql[idx].q_out = qnew
        ql[idx+1].q_in = qnew
        ql[idx+1].q_wt = qnew
        print("Cutting layer {} outputs and next layer's weights to {} bits!".format(idx, qnew))

    while not net_fits():
        for i, l in enumerate(ql):
            if not layer_fits(i):
                if i==0:
                    # first layer: just quantize outputs and next weights if
                    # applicable
                    if l.q_out > q_min and ql[i+1].q_wt > q_min:
                        cut_next(i)
                    else:
                        print("Failed to quantize first layer to meet constraints, returning original layer stack!")
                        return ql_in
                elif i < len(ql)-1:
                    # middle layers: cut inputs+current weights OR outputs +
                    # next weights
                    cur_contrib = l.in_size + l.tot_param_size
                    next_contrib = l.out_size + ql[i+1].tot_param_size
                    if cur_contrib > next_contrib:
                        if l.q_in > q_min and l.q_wt > q_min:
                            cut_cur(i)
                        # if we can't quantize current layer's input/weights,
                        # try to quantize outputs & next layer's weights
                        elif l.q_out > q_min and ql[i+1].q_wt > q_min:
                            cut_next(i)
                        else:
                            print("Failed to quantize layer {}, returning original layer stack!".format(i))
                            return ql_in
                    else:
                        if l.q_out > q_min and ql[i+1].q_wt > q_min:
                            cut_next(i)
                        elif l.q_in > q_min and l.q_wt > q_min:
                            cut_cur(i)
                        else:
                            print("Failed to quantize layer {}, returning original layer stack!".format(i))
                            return ql_in

                else:
                    # last layer - can only quantize inputs
                    if l.q_in > q_min and l.q_wt > q_min:
                        cut_cur(i)
                    else:
                        print("Failed to quantize last layer, returning original layer stack!")
                        return ql_in
        if q_min_wts is not None:
            for l in ql:
                l.q_wt = max(q_min_wts, l.q_wt)
            if not net_fits():
                print("Failed to meet minimum weight precision requirements, returning original layer stack!")
                return ql_in

    return ql


def cut_acts_wts_pulp_mp(ql_in : list, mem_rw : int, qw_min : int, qa_min : int):
    ql = copy(ql_in)

    mem_contributor = namedtuple('mem_contributor', 'kind size layer')

    def layer_size(idx : int):
        l = ql[idx]
        tot_size = l.in_size + l.out_size + l.tot_param_size
        if idx != len(ql)-1:
            tot_size += ql[idx+1].tot_param_size
        return tot_size

    def layer_fits(idx : int):
        # a layer fits if its inputs, outputs, weights and the weights of the
        # next layer all fit in RAM
        tot_size = layer_size(idx)
        return tot_size <= mem_rw

    def net_fits():
        # the net fits if all the layers fit
        return reduce(lambda a, b: a and b, [layer_fits(i) for i in range(len(ql))])

    while not net_fits():
        for i, l in enumerate(ql):
            if not layer_fits(i):
                contributors = []
                contributors.append(mem_contributor(kind='wt', size=l.tot_param_size, layer=l))
                if i != len(ql)-1:
                    contributors.append(mem_contributor(kind='wt', size=ql[i+1].tot_param_size, layer=ql[i+1]))
                contributors.append(mem_contributor(kind='in_act', size=l.in_size, layer=l))
                contributors.append(mem_contributor(kind='out_act', size=l.out_size, layer=l))
                contribs_by_size = sorted(contributors, key=lambda c: c.size)[::-1]
                for c in contribs_by_size:
                    if c.kind == 'wt':
                        if c.layer.q_wt > qw_min:
                            c.layer.q_wt = int(c.layer.q_wt/2)
                            if c.layer is l:
                                teh_layer = "current layer"
                            else:
                                teh_layer = "next layer"
                            print("Layer {}: Cutting {}'s weights to {} bits!".format(i, teh_layer, c.layer.q_wt))
                            break
                    else:
                        if c.kind == 'in_act' and i != 0:
                            if l.q_in > qa_min:
                                l.q_in = int(l.q_in/2)
                                ql[i-1].q_out = l.q_in
                                print("Layer {}: Cutting input acts to {} bits!".format(i, l.q_in))
                                break
                        elif c.kind == 'out_act' and i != len(ql)-1:
                            if l.q_out > qa_min:
                                l.q_out = int(l.q_out/2)
                                ql[i+1].q_in = l.q_out
                                print("Layer {}: Cutting output acts to {} bits!".format(i, l.q_out))
                                break
                else:
                    print("Failed to quantize layer {} - returning original layer stack!")
                    return ql_in


    return ql


def print_net_summary(ql : list):
    print("Net has {} layers:".format(len(ql)))
    ro_bytes = 0
    for i, l in enumerate(ql):
        print("Layer {}:".format(i+1), "  Type: ", "Conv" if isinstance(l, QuantConvLayer) else "Linear")
        print("  Input Dimensions:       {}".format(l.in_dims))
        if isinstance(l, QuantConvLayer):
            print("  Kernel size:          {}".format(l.conv.kernel_size))
        print("  Input Channels:         {}".format(l.conv.in_channels if isinstance(l, QuantConvLayer) else l.lin.in_features))
        print("  Output Channels:        {}".format(l.conv.out_channels if isinstance(l, QuantConvLayer) else l.lin.out_features))
        print("  Input Activation Bits:  {}".format(l.q_in))
        print("  Output Activation Bits: {}".format(l.q_out))
        print("  Weight Bits:            {}".format(l.q_wt))
        ro_bytes += l.wt_size

    print("\n\nTotal required RO mem size in bytes:")
    print("---------------------------------")
    print("|{:>27} B |".format(ro_bytes))
    print("---------------------------------")
