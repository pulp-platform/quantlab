# TODO: Write documentation

# This module generates system level stimuli and expected responses based on randomly
# generated pyTorch networks.
import os
from pathlib import Path
import shutil, glob
import json
import numpy as np
import torch
import torch.nn as nn

np.random.seed(69)
torch.manual_seed(42)

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, OrderedDict
from tqdm import tqdm
import argparse

from . import gen_activationmemory_full_stimuli as actmemory
from . import gen_weightmemory_full_stimuli as weightmemory
from . import gen_ocu_pool_weights_stimuli as ocu
from . import gen_LUCA_stimuli as LUCA
# from bitarray import bitarray

ACTMEM_START_ADDR = int("0x1EC00000",0)
WEIGHTMEM_START_ADDR = int("0x1EC40000",0)



from .global_parameters import *

numbanks = int(k * weight_stagger)

totnumtrits = imagewidth * imageheight * ni
tritsperbank = int(np.ceil(totnumtrits / numbanks))

effectivetritsperword = int(ni / weight_stagger)
physicaltritsperword = int(np.ceil(effectivetritsperword / 5)) * 5
physicalbitsperword = int(physicaltritsperword / 5 * 8)
excessbits = (physicaltritsperword - effectivetritsperword) * 2
effectivewordwidth = physicalbitsperword - excessbits
numdecoders = int(physicalbitsperword / 8)

bankdepth = int(np.ceil(tritsperbank / effectivetritsperword))

bankaddressdepth = int(np.ceil(np.log2(bankdepth)))

leftshiftbitwidth = int(np.ceil(np.log2(numbanks)))
splitbitwidth = int(np.ceil(np.log2(weight_stagger))) + 1

nibitwidth = int(np.maximum(np.ceil(np.log2(ni)), 1)) + 1
nobitwidth = int(np.maximum(np.ceil(np.log2(no)), 1))
imagewidthbitwidth = int(np.maximum(np.ceil(np.log2(imagewidth)), 1)) + 1
imageheightbitwidth = int(np.maximum(np.ceil(np.log2(imageheight)), 1)) + 1

numaddresses = int(numbanks * bankdepth)
memaddressbitwidth = int(np.maximum(np.ceil(np.log2(numaddresses)), 1))

leftshiftbitwidth = int(np.ceil(np.log2(numbanks)))
splitbitwidth = int(np.ceil(np.log2(weight_stagger))) + 1

rowaddresswidth = int(np.ceil(np.log2(imw)))
coladdresswidth = int(np.ceil(np.log2(imagewidth)))
tcnwidthaddrwidth = int(np.ceil(np.log2(tcn_width)))

matrixaddresswidth = int(np.ceil(np.log2(imageheight * imagewidth))) + 1
kaddresswidth = int(np.ceil(np.log2(k)))

_output = namedtuple("_outputs", "actmemory_external_acts_o")
_input = namedtuple("_inputs",
                    "actmemory_external_bank_set actmemory_external_we actmemory_external_req actmemory_external_addr actmemory_external_wdata weightmemory_external_bank weightmemory_external_we weightmemory_external_req weightmemory_external_addr weightmemory_external_wdata ocu_thresh_pos ocu_thresh_neg ocu_thresholds_save_enable LUCA_store_to_fifo LUCA_testmode LUCA_imagewidth LUCA_imageheight LUCA_k LUCA_ni LUCA_no LUCA_stride_width LUCA_stride_height LUCA_padding_type LUCA_pooling_enable LUCA_pooling_pooling_type LUCA_pooling_kernel LUCA_pooling_padding_type LUCA_layer_skip_in LUCA_layer_skip_out LUCA_layer_is_tcn LUCA_layer_tcn_width_mod_dil LUCA_layer_tcn_k LUCA_compute_disable")

outputtypes = _output("unsigned")
inputtypes = _input(actmemory.inputtypes.external_bank_set, actmemory.inputtypes.external_we,
                    actmemory.inputtypes.external_req, actmemory.inputtypes.external_addr,
                    actmemory.inputtypes.external_wdata, "unsigned", weightmemory.inputtypes.external_we,
                    weightmemory.inputtypes.external_req, weightmemory.inputtypes.external_addr,
                    weightmemory.inputtypes.external_wdata, ocu.inputtypes.thresh_pos, ocu.inputtypes.thresh_neg,
                    ocu.inputtypes.threshold_store_to_fifo, LUCA.inputtypes.store_to_fifo, LUCA.inputtypes.testmode,
                    LUCA.inputtypes.imagewidth, LUCA.inputtypes.imageheight, LUCA.inputtypes.k, LUCA.inputtypes.ni,
                    LUCA.inputtypes.no, LUCA.inputtypes.stride_width, LUCA.inputtypes.stride_height,
                    LUCA.inputtypes.padding_type, LUCA.inputtypes.pooling_enable, LUCA.inputtypes.pooling_pooling_type,
                    LUCA.inputtypes.pooling_kernel, LUCA.inputtypes.pooling_padding_type, LUCA.inputtypes.skip_in,
                    LUCA.inputtypes.skip_out, "unsigned", "unsigned", "unsigned", LUCA.inputtypes.compute_disable)

outputwidths = _output((physicalbitsperword, (1)))
inputwidths = _input(actmemory.inputwidths.external_bank_set, actmemory.inputwidths.external_we,
                     actmemory.inputwidths.external_req, actmemory.inputwidths.external_addr,
                     actmemory.inputwidths.external_wdata, (nobitwidth, 1), weightmemory.inputwidths.external_we,
                     weightmemory.inputwidths.external_req, weightmemory.inputwidths.external_addr,
                     weightmemory.inputwidths.external_wdata, ocu.inputwidths.thresh_pos, ocu.inputwidths.thresh_neg,
                     ocu.inputwidths.threshold_store_to_fifo, LUCA.inputwidths.store_to_fifo, LUCA.inputwidths.testmode,
                     LUCA.inputwidths.imagewidth, LUCA.inputwidths.imageheight, LUCA.inputwidths.k, LUCA.inputwidths.ni,
                     LUCA.inputwidths.no, LUCA.inputwidths.stride_width, LUCA.inputwidths.stride_height,
                     LUCA.inputwidths.padding_type, LUCA.inputwidths.pooling_enable,
                     LUCA.inputwidths.pooling_pooling_type, LUCA.inputwidths.pooling_kernel,
                     LUCA.inputwidths.pooling_padding_type, LUCA.inputwidths.skip_in, LUCA.inputwidths.skip_out,
                     (1, (1)), (coladdresswidth, (1)), (kaddresswidth, (1)), LUCA.inputwidths.compute_disable)

_layer_param = namedtuple("_layer_params", "imagewidth "
                                           "imageheight "
                                           "k "
                                           "ni "
                                           "no "
                                           "stride_width "
                                           "stride_height "
                                           "padding_type "
                                           "pooling_enable "
                                           "pooling_pooling_type "
                                           "pooling_kernel "
                                           "pooling_padding_type "
                                           "skip_in "
                                           "skip_out "
                                           "is_tcn "
                                           "tcn_width "
                                           "tcn_width_mod_dil "
                                           "tcn_k ")

layer_param_types = _layer_param(imagewidth=LUCA.inputtypes.imagewidth,
                                 imageheight=LUCA.inputtypes.imageheight,
                                 k=LUCA.inputtypes.k,
                                 ni=LUCA.inputtypes.ni,
                                 no=LUCA.inputtypes.no,
                                 stride_width=LUCA.inputtypes.stride_width,
                                 stride_height=LUCA.inputtypes.stride_height,
                                 padding_type=LUCA.inputtypes.padding_type,
                                 pooling_enable=LUCA.inputtypes.pooling_enable,
                                 pooling_pooling_type=LUCA.inputtypes.pooling_pooling_type,
                                 pooling_kernel=LUCA.inputtypes.pooling_kernel,
                                 pooling_padding_type=LUCA.inputtypes.pooling_padding_type,
                                 skip_in=LUCA.inputtypes.skip_in,
                                 skip_out=LUCA.inputtypes.skip_out,
                                 is_tcn='unsigned',
                                 tcn_width='unsigned',
                                 tcn_width_mod_dil='unsigned',
                                 tcn_k='unsigned')

layer_param_widths = _layer_param(imagewidth=LUCA.inputwidths.imagewidth,
                                  imageheight=LUCA.inputwidths.imageheight,
                                  k=LUCA.inputwidths.k,
                                  ni=LUCA.inputwidths.ni,
                                  no=LUCA.inputwidths.no,
                                  stride_width=LUCA.inputwidths.stride_width,
                                  stride_height=LUCA.inputwidths.stride_height,
                                  padding_type=LUCA.inputwidths.padding_type,
                                  pooling_enable=LUCA.inputwidths.pooling_enable,
                                  pooling_pooling_type=LUCA.inputwidths.pooling_pooling_type,
                                  pooling_kernel=LUCA.inputwidths.pooling_kernel,
                                  pooling_padding_type=LUCA.inputwidths.pooling_padding_type,
                                  skip_in=LUCA.inputwidths.skip_in,
                                  skip_out=LUCA.inputwidths.skip_out,
                                  is_tcn=(1, (1)),
                                  tcn_width=(tcnwidthaddrwidth, (1)),
                                  tcn_width_mod_dil=(tcnwidthaddrwidth, (1)),
                                  tcn_k=(kaddresswidth, (1)))

_weightmem_writes = namedtuple("_weightmem_writes", "addr bank wdata")
weightmem_writes_types = _weightmem_writes(addr='unsigned',
                                           bank='unsigned',
                                           wdata='unsigned')
weightmem_writes_widths = _weightmem_writes(addr=weightmemory.inputwidths.external_addr,
                                            bank=(nobitwidth, 1),
                                            wdata=weightmemory.inputwidths.external_wdata)

_thresholds = namedtuple("_thresholds", "pos neg we")
thresholds_types = _thresholds(pos='signed',
                               neg='signed',
                               we='unsigned')
thresholds_widths = _thresholds(pos=ocu.inputwidths.thresh_pos,
                                neg=ocu.inputwidths.thresh_neg,
                                we=(1, (ni)))

Thresholds = namedtuple('Thresholds', 'lo hi')

cyclenum = 0

pipelinedelay = 1
widthcounter = 0
heightcounter = 0
counting = 1

actmemory.codebook, _ = actmemory.gen_codebook()
reverse_codebook = {}

for x, y in actmemory.codebook.items():
    if y not in reverse_codebook:
        reverse_codebook[y] = x

def format_output(output):
    string = ''

    for i in range(output.shape[0]):
        for k in range(output.shape[2]):
            for l in range(output.shape[3]):
                for j in range(output.shape[1]):
                    string += (format_ternary(output[i][j][k][l])) + ' '
                string = ''

def format_signals(_signals, signaltypes, signalwidths):
    string = ""

    for j in _signals._fields:
        for i in np.nditer(getattr(_signals, j)):
            string = string + _format(i, getattr(signaltypes, j), (getattr(signalwidths, j))[0])

    return string

def _format(num, _type, bitwidth=1):
    if (_type == 'ternary'):
        return format_ternary(num)
    else:
        return format_binary(num, bitwidth)

def format_ternary(num):
    if (num == 1):
        return '01';
    elif (num == 0):
        return '00';
    elif (num == -1):
        return '11';
    else:
        return 'XX';

def format_binary(num, bitwidth):
    max_val = int(2 ** (bitwidth))

    if (num < 0):
        neg_flag = 1
    else:
        neg_flag = 0

    if (neg_flag == 1):
        _num = max_val + num
    else:
        _num = num

    string = bin(int(_num))[2:]

    string = string.zfill(bitwidth)

    return str(string[-bitwidth:])

def get_thresholds(conv_node, bn_node):
    beta_hat = (conv_node.bias - bn_node.running_mean) / torch.sqrt(bn_node.running_var + bn_node.eps)
    gamma_hat = 1 / torch.sqrt(bn_node.running_var + bn_node.eps)
    beta_hat = beta_hat * bn_node.weight + bn_node.bias
    gamma_hat = gamma_hat * bn_node.weight

    thresh_high = (0.5 - beta_hat) / gamma_hat
    thresh_low = (-0.5 - beta_hat) / gamma_hat

    flip_idxs = gamma_hat < 0
    thresh_high[flip_idxs] *= -1
    thresh_low[flip_idxs] *= -1
    thresh_high = torch.ceil(thresh_high)
    thresh_low = torch.ceil(thresh_low)
    thresh_low = torch.where(torch.eq(thresh_low, -0.), torch.zeros_like(thresh_low), thresh_low)
    thresh_high = torch.where(torch.eq(thresh_high, -0.), torch.zeros_like(thresh_high), thresh_high)
    return Thresholds(thresh_low, thresh_high)


def double_threshold(x, xmin, xmax):
    if x.ndim == 4:
        xmin = xmin.unsqueeze(-1).unsqueeze(-1)
        xmax = xmax.unsqueeze(-1).unsqueeze(-1)
    elif x.ndim == 3:
        xmax = xmax.unsqueeze(-1)
        xmin = xmin.unsqueeze(-1)

    max_t = torch.gt(x, xmax)
    min_t = torch.gt(-x, -xmin) * (-1)

    return (max_t + min_t).float()

class DensetoConv(nn.Module):
    def __init__(self, input_shape, n_classes, weights):
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.rounded_channels = int(np.ceil(input_shape[1] / (ni // weight_stagger)) * ni // weight_stagger)
        self.rounded_weights = torch.zeros(n_classes, self.rounded_channels, input_shape[-1])
        self.rounded_weights[:,:input_shape[1]] = torch.Tensor(weights)
        self.n_inputs = int(np.prod(weights.shape[1:]))
        with torch.no_grad():
            super(DensetoConv, self).__init__()
            self.dense = nn.Linear(self.n_inputs, out_features=n_classes)
            self.dense.weight = nn.Parameter(torch.Tensor(weights).view(n_classes, -1))
            self.dense.bias = nn.Parameter(torch.zeros_like(self.dense.bias))
            self.thresh = Thresholds(0., 0.)
            self.conv = nn.Conv2d(in_channels=self.rounded_channels,
                                  out_channels=n_classes,
                                  kernel_size=k,
                                  bias=False)
            self.conv.weight.copy_(self.weights_to_conv(weights=self.dense.weight))
            self.conv.weight.requires_grad = False

    def acts_to_conv(self, acts):
        conv_acts = torch.zeros(1, self.rounded_channels, k, k)

        # print('acts', acts.shape, 'reshaped to', conv_acts.shape)
        for i in range(acts.shape[1]):
            n_pixels = np.prod(self.input_shape[2:])
            x = (i % n_pixels) % k
            y = (i % n_pixels) // k
            c = i // (n_pixels)
            conv_acts[0][c][y][x] = acts[0][i]
        return conv_acts

    def weights_to_conv(self, weights):
        conv_weights = torch.zeros(self.n_classes, self.rounded_channels, k, k)

        print('weights', weights.shape, 'reshaped to', conv_weights.shape)
        for o in range(self.n_classes):
            for i in range(weights.shape[1]):

                n_pixels = np.prod(self.input_shape[2:])
                x = (i % n_pixels) % k
                y = (i % n_pixels) // k
                c = i // (n_pixels)
                conv_weights[o][c][y][x] = weights[o][i]
        return conv_weights

    def forward(self, x):
        x_pad = torch.zeros(1, self.rounded_channels, self.input_shape[-1])
        x_pad[:,:self.input_shape[1]] = x.view(self.input_shape)
        y = self.conv(self.acts_to_conv(x_pad.view(1, -1)))
        x = self.dense(x)
        assert torch.equal(y.squeeze(), x.squeeze())
        return x



class Net(nn.Module):
    def __init__(self, num_cnn_layers, num_tcn_layers, num_dense_layers, layer_no, layer_ni, n_classes, layer_k, strideh, stridew,
                 layer_padding, pooling_enable, pooling_type, pooling_kernel, pooling_padding_type, tcn_k, tcn_dilation,
                 tcn_width, imagewidth, imageheight,weights, thresh_neg, thresh_pos):
        super(Net, self).__init__()
        self.cnns = nn.ModuleList()
        self.tcns = nn.ModuleList()
        self.dense = None
        self.tcn_sequence = None
        self.cnn_thresh = []
        self.tcn_thresh = []
        # CNN Layers
        for i in range(num_cnn_layers):
            with torch.no_grad():
                # Convolution
                conv = nn.Conv2d(in_channels=layer_ni[i],
                                 out_channels=layer_no[i],
                                 padding=(layer_k[i] - 1) // 2 * layer_padding[i],
                                 kernel_size=layer_k[i],
                                 stride=(strideh[i], stridew[i]))

                conv.weight = nn.Parameter(torch.Tensor(weights[i]))
                conv.bias = nn.Parameter(torch.zeros_like(conv.bias))
                conv.weight.requires_grad = False
                # Pooling
                if pooling_enable[i]:
                    pool = pooling_type[i](kernel_size=pooling_kernel[i], padding=pooling_padding_type)
                else:
                    pool = nn.Identity()
                self.cnn_thresh.append(Thresholds(lo=torch.Tensor(thresh_neg[i]), hi=torch.Tensor(thresh_pos[i])))
                self.cnns.append(nn.Sequential(OrderedDict([('conv', conv), ('pool', pool)])))

        # TCN Layers
        for i in range(num_tcn_layers):
            with torch.no_grad():
                if i == 0:
                    self.tcn_sequence = torch.zeros((1, layer_ni[num_cnn_layers + i], tcn_width[i]))
                # Left Padding
                padding = nn.ConstantPad1d(padding=((tcn_k[i] - 1) * tcn_dilation[i], 0), value=0.)
                # Convolution
                conv = nn.Conv1d(in_channels=layer_ni[num_cnn_layers + i],
                                 out_channels=layer_no[num_cnn_layers + i],
                                 kernel_size=tcn_k[i],
                                 dilation=tcn_dilation[i])
                conv.weight = nn.Parameter(torch.Tensor(weights[num_cnn_layers + i]))
                conv.bias = nn.Parameter(torch.zeros_like(conv.bias))
                conv.weight.requires_grad = False
                # Thresholds
                self.tcn_thresh.append(Thresholds(lo=torch.Tensor(thresh_neg[num_cnn_layers + i]), hi=torch.Tensor(thresh_pos[num_cnn_layers + i])))
                self.tcns.append(nn.Sequential(OrderedDict([('pad', padding), ('conv', conv)])))

        # Dense Layer
        if num_dense_layers == 1:
            x = torch.zeros(1, layer_ni[0], imagewidth, imageheight)
            for i in self.cnns:
                x = i.conv(x)
                x = i.pool(x)
            x = torch.flatten(x, start_dim=1).squeeze()
            self.tcn_sequence[:, :, :-1] = self.tcn_sequence[:, :, 1:].clone()
            self.tcn_sequence[:, :, -1] = x
            x = self.tcn_sequence
            for i in self.tcns:
                x = i.pad(x)
                x = i.conv(x)
            self.dense = DensetoConv(x.shape, n_classes, weights[-1])

        #print(self)

    def forward(self, x):
        shapes = [x.shape]

        # CNN forward
        for cnn, thresh in zip(self.cnns, self.cnn_thresh):
            # I am abusing batch size as tcn_width
            x = cnn.conv(x)
            x = cnn.pool(x)
            x = double_threshold(x, xmin=thresh.lo, xmax=thresh.hi)
            shapes.append(x.shape)

        # TCN forward
        if self.tcns:
            # input shape to TCN is (1, channels, width, height)
            # -> reshape it to (channels)

            x = torch.flatten(x, start_dim=1).squeeze()
            self.tcn_sequence[:,:,:-1] = self.tcn_sequence[:,:,1:].clone()
            self.tcn_sequence[:,:,-1] = x
            x = self.tcn_sequence
            shapes[-1] = x.shape
            for tcn, thresh in zip(self.tcns, self.tcn_thresh):
                x = tcn.pad(x)
                x = tcn.conv(x)
                x = double_threshold(x, xmin=thresh.lo, xmax=thresh.hi)
                shapes.append(x.shape)

        # Dense forward
        if self.dense:
            x = torch.flatten(x, start_dim=1)
            shapes[-1] = x.shape
            x = self.dense(x)
            # x = double_threshold(x, self.dense.thresh.lo, self.dense.thresh.hi)
            shapes.append(x.shape)
            x = x.unsqueeze(-1)
        return x, shapes

    def reset(self):
        if torch.is_tensor(self.tcn_sequence):
            self.tcn_sequence = torch.zeros_like(self.tcn_sequence)



def make_random_image(imagewidth, imageheight, layer_ni, rounded_ni):
    zero_pad_image = torch.zeros((1, rounded_ni, imagewidth, imageheight))
    actual_image = torch.randint(-1, 2, (1, layer_ni, imagewidth, imageheight), dtype=torch.float32)
    zero_pad_image[0, :layer_ni] = actual_image
    return actual_image, zero_pad_image

def make_random_tcn_sequence(net, layer_ni, rounded_ni, length):
    testsequence = np.zeros((1, rounded_ni, length, 1))
    testsequence[0,:layer_ni,:,0] = np.random.randint(-1, 2, (layer_ni, length))
    return testsequence

def translate_tcn_weights_to_cnn_weights(weights):
    weights2D = np.zeros((*weights.shape[:2], k, k))
    weights2D[:,:,:weights.shape[2], (k-1)//2] = weights
    return weights2D

def translate_weights_to_weightmem(weights, i=-1, cnn=False):
    n_o, n_i, k1, k2 = weights.shape
    rounded_no = int(np.ceil(n_o / effectivetritsperword) * effectivetritsperword)
    #rounded_no = no
    #if cnn and i==0:
    #    rounded_ni = int(np.ceil(n_i / effectivetritsperword) * effectivetritsperword)
    #else:
    #    rounded_ni = ni
    rounded_ni = int(np.ceil(n_i / effectivetritsperword) * effectivetritsperword)
    zero_padded_weights = np.zeros((rounded_no, rounded_ni, k1, k2))
    zero_padded_weights[:n_o, :n_i] = weights

    weights = zero_padded_weights

    weightmem = np.empty((int(np.prod(weights.shape) / (ni / weight_stagger)), physicalbitsperword), dtype=int)
    weightmem_decoded = np.empty((int(np.prod(weights.shape) / (ni / weight_stagger)), effectivetritsperword*2), dtype=int)
    weightmemlist, weightmemlist_decoded = [], []
    for i in range(weights.shape[0]):
        for n in range(int(weights.shape[1] / int(ni / weight_stagger))):
            for m in range(weights.shape[3]):
                for j in range(weights.shape[2]):
                    word = np.empty(int(ni / weight_stagger))
                    for q in range(int(ni / weight_stagger)):
                        word[q] = weights[i][n * int(ni / weight_stagger) + q][j][m]
                    _word, word_decoded = translate_ternary_sequence(word)
                    weightmemlist_decoded.append(translate_binary_string(word_decoded))
                    weightmemlist.append(translate_binary_string(_word))
    weightmemarray = np.asarray(weightmemlist)
    weightmemarray_decoded = np.asarray(weightmemlist_decoded)
    weightmem = weightmemarray.reshape((int(np.prod(weights.shape) / (ni / weight_stagger)), physicalbitsperword))
    return weightmem, weightmemarray_decoded

def translate_image_to_actmem(image):
    image = image.unsqueeze(-1) if image.ndim == 3 else image
    actmem = np.empty(
        (int(np.ceil((image.shape[1]) / weight_stagger)) * image.shape[2] * image.shape[3], physicalbitsperword),
        dtype=int)
    actmemlist, actmemlist_decoded = [], []

    for n in range(image.shape[2]):
        for m in range(image.shape[3]):
            for j in range(int(np.ceil((image.shape[1]) / (ni / weight_stagger)))):
                word = np.empty(int(ni / weight_stagger))
                for i in range(min(image.shape[1] - j * int((ni / weight_stagger)), int(ni / weight_stagger))):
                    word[i] = image[0][i + j * int((ni / weight_stagger))][n][m]
                _word, word_decoded = translate_ternary_sequence(word)
                actmemlist.append(translate_binary_string(_word))
                actmemlist_decoded.append(translate_binary_string(word_decoded))
    actmemarray = np.asarray(actmemlist)
    actmemarray_decoded = np.asarray(actmemlist_decoded)
    actmem = actmemarray.reshape((-1, physicalbitsperword))

    return actmem, actmemarray_decoded

def translate_binary_string(string):
    ret = np.empty(len(string), dtype=int)
    for i in range(len(string)):
        ret[i] = string[i]

    return (ret)

def translate_ternary_sequence(seq):
    string = ''
    _seq = np.copy(seq.reshape(-1))

    for i in range(len(_seq)):
        if (int(_seq[i]) == 1):
            string += "01"
        elif (int(_seq[i]) == -1):
            string += "11"
        else:
            string += "00"

    string_decoded = string
    string += "0000"
    _string = ''
    for i in range(0, int(len(string)), 10):
        substr = string[i:i + 10]
        _string += reverse_codebook[substr]

    return _string, string_decoded

def compute_tnn(n_layers : int, in_directory : str, in_prefix : str, out_directory : str, n_windows : int = None):

    num_layers = len([f for f in os.listdir(in_directory) if '.json' in f]) if n_layers == 0 else n_layers
    num_cnn_layers, num_tcn_layers, num_dense_layers = 0, 0, 0


    weights_f = []
    thresholds_lo_f, thresholds_hi_f = [], []

    layer_pooling_enable = []
    layer_pooling_type = []
    layer_pooling_padding_type = []
    layer_pooling_kernel = []
    layer_strideh = []
    layer_stridew = []
    layer_padding = []
    layer_channels = []

    layer_k = []
    # layer_padding = 1

    layer_tcn_width = []
    layer_tcn_dilation = []
    layer_tcn_k = []

    for l in range(num_layers):
        # layer configs
        layer_cfg_f = open(os.path.join(in_directory, '{}_l{}_config.json'.format(in_prefix, l)), 'r')
        layer_cfg = json.load(layer_cfg_f)
        layer_channels.append(layer_cfg['conv_in_ch'])

        if layer_cfg['conv_type'] == '2d':
            num_cnn_layers += 1
            layer_k.append(layer_cfg['conv_k'][0])
            layer_pooling_enable.append(layer_cfg['pooling'])
            layer_pooling_type.append(nn.MaxPool2d if layer_cfg['pool_type'] == 'max' else nn.AvgPool2d)
            layer_pooling_padding_type.append(0)
            layer_pooling_kernel.append(layer_cfg['pool_k'][0])
            layer_strideh.append(layer_cfg['conv_stride'][0])
            layer_stridew.append(layer_cfg['conv_stride'][1])
            layer_padding.append(layer_cfg['conv_padding'])
            # layer thresholds
            thresholds_lo_f.append(np.clip(np.load(os.path.join(in_directory, '{}_l{}_thresh_lo.npy'.format(in_prefix, l))), -(2 ** (thresholds_widths.pos[0]-1)), (2 ** (thresholds_widths.pos[0]-1)) - 1))
            thresholds_hi_f.append(np.clip(np.load(os.path.join(in_directory, '{}_l{}_thresh_hi.npy'.format(in_prefix, l))), -(2 ** (thresholds_widths.pos[0]-1)), (2 ** (thresholds_widths.pos[0]-1)) - 1))

        elif layer_cfg['conv_type'] == '1d' and not layer_cfg['fp_out']:
            num_tcn_layers += 1
            layer_k.append(3)
            layer_tcn_k.append(layer_cfg['conv_k'])
            layer_tcn_width.append(layer_cfg['n_tcn_steps'])
            layer_tcn_dilation.append(layer_cfg['dilation'])
            # layer thresholds
            thresholds_lo_f.append(np.clip(np.load(os.path.join(in_directory, '{}_l{}_thresh_lo.npy'.format(in_prefix, l))), -(2 ** (thresholds_widths.pos[0]-1)), (2 ** (thresholds_widths.pos[0]-1)) - 1))
            thresholds_hi_f.append(np.clip(np.load(os.path.join(in_directory, '{}_l{}_thresh_hi.npy'.format(in_prefix, l))), -(2 ** (thresholds_widths.pos[0]-1)), (2 ** (thresholds_widths.pos[0]-1)) - 1))
        else:
            num_dense_layers += 1
            layer_k.append(3)
            # thresholds not needed! just here to fill up the threshold fifo
            thresholds_lo_f.append(np.zeros(layer_cfg['conv_out_ch']))
            thresholds_hi_f.append(np.zeros(layer_cfg['conv_out_ch']))
    

        # layer weights
        weights_f.append(np.load(os.path.join(in_directory, '{}_l{}_weights.npy'.format(in_prefix, l))))
    # append last output channel
    layer_channels.append(layer_cfg['conv_out_ch'])
    if not (n_layers > 0 and n_layers < 6) and n_windows is None:
        n_wins = layer_tcn_width[-1]

    elif not (n_layers > 0 and n_layers < 6):
        n_wins = n_windows
    else:
        n_wins = 1

    num_execs = n_wins
    # specify input dim
    input_f = np.load(os.path.join(in_directory, '{}_input_0.npy'.format(in_prefix)))
    _, input_imagewidth, input_imageheight = input_f.shape

    #print(layer_channels, layer_pooling_enable)

    n_classes = layer_channels[-1]
    layer_ni = layer_channels[:-1]
    layer_no = layer_channels[1:]

    assert len(layer_channels) == num_layers + 1
    assert len(layer_strideh) >= num_cnn_layers
    assert len(layer_stridew) >= num_cnn_layers
    assert len(layer_tcn_k) >= num_tcn_layers
    assert len(layer_tcn_dilation) >= num_tcn_layers
    assert len(layer_pooling_enable) >= num_cnn_layers



    rounded_no = [int(np.ceil(n / (ni // weight_stagger)) * ni // weight_stagger) for n in layer_no]
    rounded_ni = [int(np.ceil(n / (ni // weight_stagger)) * ni // weight_stagger) for n in layer_ni]

    net = Net(num_cnn_layers=num_cnn_layers,
              num_tcn_layers=num_tcn_layers,
              num_dense_layers=num_dense_layers,
              layer_ni=layer_ni,
              layer_no=layer_no,
              n_classes=n_classes,
              layer_k=layer_k,
              stridew=layer_stridew,
              strideh=layer_strideh,
              layer_padding=layer_padding,
              pooling_enable=layer_pooling_enable,
              pooling_type=layer_pooling_type,
              pooling_kernel=layer_pooling_kernel,
              pooling_padding_type=0,
              tcn_k=layer_tcn_k,
              tcn_dilation=layer_tcn_dilation,
              tcn_width=layer_tcn_width,
              imagewidth=input_imagewidth,
              imageheight=input_imageheight,
              weights=weights_f,
              thresh_neg=thresholds_lo_f,
              thresh_pos=thresholds_hi_f)



    weightmem_layers, weightmem_layers_decoded = [], []
    #print(f"num_cnn_layers: {num_cnn_layers}")
    for i in range(num_cnn_layers):
        weights_i = weights_f[i]
        weightmem_layer, weightmem_layer_decoded = translate_weights_to_weightmem(weights_i, i, True)
        weightmem_layers.append(weightmem_layer)
        weightmem_layers_decoded.append(weightmem_layer_decoded)
    for i in range(num_tcn_layers):
        weightmem_layer, weightmem_layer_decoded = translate_weights_to_weightmem(translate_tcn_weights_to_cnn_weights(net.tcns[i].conv.weight))
        weightmem_layers.append(weightmem_layer)
        weightmem_layers_decoded.append(weightmem_layer_decoded)
    for i in range(num_dense_layers):
        weightmem_layer, weightmem_layer_decoded = translate_weights_to_weightmem(net.dense.conv.weight)
    
        weightmem_layers.append(weightmem_layer)
        weightmem_layers_decoded.append(weightmem_layer_decoded)

    weightmem, weightmem_decoded = np.concatenate(weightmem_layers), np.concatenate(weightmem_layers_decoded)

    net.reset()
    all_results = []
    all_inputs = []
    for i in range(n_wins):
        input_f = np.load(os.path.join(in_directory, f'{in_prefix}_input_{i}.npy'))
        all_inputs.append(input_f)
        result, outshapes = net(torch.Tensor(input_f).unsqueeze(0))
        all_results.append(result)
        print(f"result {i} in compute_dvstnn:")
        print(result.detach().squeeze().numpy())

    #for i, r in enumerate(all_results):
#        print(f"result {i+1}:\n{r}")

    # input_f = np.load(os.path.join(in_directory, f'{}_input_{i}.npy'))
    # result, outshapes = net(torch.Tensor(input_f).unsqueeze(0))
    # net.reset()
    # for i in outshapes:
    #     print(i)

    num_responses = np.prod(outshapes[-1][-2:])*layer_no[-1]/(no//weight_stagger)
    num_acts = np.prod(outshapes[0][-2:])*layer_no[-1]/(no//weight_stagger)


    weightmemorywrites = 0
    threshold_writes = 0
    memwrites = []
    actmemorywrites = int(np.ceil(rounded_ni[0] / (ni / weight_stagger)) * input_imagewidth * input_imageheight)
    for i in range(num_layers):
        weightmemorywrites += int(np.ceil(rounded_ni[i] / (ni / weight_stagger)) * layer_k[i] * layer_k[i] * rounded_no[i])
        threshold_writes += layer_no[i]
        memwrites.append({'layer': i, 'ni': rounded_ni[i], 'no': rounded_no[i], 'weight_writes': weightmemorywrites,
                          'thresh_writes': threshold_writes})
    numwrites = np.maximum(actmemorywrites, weightmemorywrites)

    #for i in memwrites:
#        print(i)

    current_weight_write_layer, current_thresh_write_layer = 0, 0
    thresh_addr = 0
    weightmem_counter = 0
    weightmem_depth = np.zeros(no, dtype=int)
    weightmem_show = np.zeros((no, ni // (ni // weight_stagger) * k * k * 9))

    print("Generating layer params stimuli file...")
    #f_layer_param_intf = open("layer_params_intf.txt", 'w+')

    f_layer_param_c = open(os.path.join(out_directory,'layer_params_intf.h'), 'w+')
    f_layer_param_c.write('#ifndef __LAYER_PARAMS_INCLUDE_GUARD\n')
    f_layer_param_c.write('#define __LAYER_PARAMS_INCLUDE_GUARD\n\n')
    f_layer_param_c.write('int32_t cutieLayerParamsLen = {};\n'.format(num_layers*4))
    f_layer_param_c.write('int32_t cutieLayerParams[] PI_L2 = {\n')

    layer_param_str = ''

    for i in range(num_layers):
        #CNN Layers
        if (i < num_cnn_layers):
            b, c, h, w = outshapes[i]
            imagewidth = h
            imageheight = w
            stride_height = layer_strideh[i]
            stride_width = layer_stridew[i]
            padding_type = layer_padding[i][0]
            is_tcn = 0
            tcn_k = 0
            tcn_width_mod_dil = 0
            tcn_width = 1
            pooling_enable = layer_pooling_enable[i]
            pooling_type = int(layer_pooling_type[i] != nn.MaxPool2d)
            pooling_kernel = layer_pooling_kernel[i]
            pooling_padding_type = layer_pooling_padding_type[i]

        # TCN Layers
        elif i < num_layers - num_dense_layers:
            b, c, l = outshapes[i]
            is_tcn = 1
            tcn_k = layer_tcn_k[i - num_cnn_layers]
            tcn_width = layer_tcn_width[i - num_cnn_layers]
            imagewidth = layer_tcn_dilation[i - num_cnn_layers]
            imageheight = int(np.ceil(l / imagewidth)) + (tcn_k - 1)
            stride_width = 1
            stride_height = 1
            padding_type = 1
            pooling_enable=0
            pooling_type=0
            pooling_kernel=0
            pooling_padding_type=0
            tcn_width_mod_dil = l % imagewidth  # not dilation but modulo, because of longest path
            tcn_1d_width = l

        # Dense Layers
        else:
            imagewidth = layer_k[i]
            imageheight = layer_k[i]
            stride_height = 1
            stride_width = 1
            padding_type = 0
            is_tcn = 0
            tcn_k = 0
            tcn_width_mod_dil = 0
            pooling_enable = 0
            pooling_type = 0
            pooling_kernel = 0
            pooling_padding_type = 0

        layer_params = _layer_param(imagewidth=imagewidth,
                                    imageheight=imageheight,
                                    k=layer_k[i],
                                    ni=rounded_ni[i],
                                    no=rounded_no[i],
                                    stride_height=stride_height,
                                    stride_width=stride_width,
                                    padding_type=padding_type,
                                    pooling_enable=pooling_enable,
                                    pooling_pooling_type=pooling_type,
                                    pooling_kernel=pooling_kernel,
                                    pooling_padding_type=pooling_padding_type,
                                    skip_in=0,
                                    skip_out=0,
                                    is_tcn=is_tcn,
                                    tcn_width=tcn_width,
                                    tcn_width_mod_dil=tcn_width_mod_dil,
                                    tcn_k=tcn_k)
        #f_layer_param_intf.write("%s\n" % ",".join([str(j) for j in list(layer_params)]))

        layer_param_str += str('0x%08X,' % ( (rounded_no[i] * (2**24)) + (rounded_ni[i]*(2**16)) + (imageheight*(2**8)) + imagewidth ))
        layer_param_str += str('0x%08X,' % ( (tcn_k * (2**24)) + (tcn_width_mod_dil*(2**16)) + (tcn_width*(2**8)) + is_tcn ))
        layer_param_str += str('0x%08X,' % ( (stride_height * (2**24)) + (stride_width*(2**16)) + (padding_type*(2**8)) + layer_k[i] ))
        layer_param_str += str('0x%08X,\n' % ( (pooling_padding_type * (2**24)) + (pooling_kernel*(2**16)) + (pooling_type*(2**8)) + pooling_enable ))
    #import ipdb; ipdb.set_trace()

    f_layer_param_c.write(layer_param_str[:-2])
    #f_layer_param_intf.close()
    f_layer_param_c.write('};\n')
    f_layer_param_c.write('#endif')
    f_layer_param_c.close()
    
    print("Generating weights stimuli file...")
    #f_weightmem_writes_intf = open("weights_intf.txt", 'w+')

    f_weightmem_writes_c = open(os.path.join(out_directory, 'weights_intf.h'), 'w')
    f_weightmem_writes_c.write('#ifndef __WEIGHTS_INCLUDE_GUARD\n')
    f_weightmem_writes_c.write('#define __WEIGHTS_INCLUDE_GUARD\n\n')
    f_weightmem_writes_c.write('int32_t cutieWeightsLen = {};\n'.format(weightmemorywrites*4))
    f_weightmem_writes_c.write('int32_t cutieWeights[] PI_L2 = {\n')

    weightmem_writes_str = ''

    for i in range(weightmemorywrites):
        if i >= memwrites[current_weight_write_layer]['weight_writes']:
            weightmem_counter = 0
            current_weight_write_layer += 1
            weightmem_depth[:] = current_weight_write_layer * k * k * weight_stagger
            # weightmem_depth[:] = weightmem_depth[0]

        weightmemory_writedepth = int(layer_k[current_weight_write_layer] * layer_k[current_weight_write_layer] * np.ceil(memwrites[current_weight_write_layer]['ni'] / (ni / weight_stagger)))
        weightmemory_bank = (int(weightmem_counter / weightmemory_writedepth) % memwrites[current_weight_write_layer]['no'])
        weightmemory_addr = weightmem_depth[weightmemory_bank]
        weightmem_depth[weightmemory_bank] += 1
        weightmemory_wdata = weightmem[i]
        weightmem_show[weightmemory_bank, weightmemory_addr] = i  # current_weight_write_layer + 1
        weightmem_counter += 1
        weights = _weightmem_writes(addr=weightmemory_addr,
                                    bank=weightmemory_bank,
                                    wdata=weightmemory_wdata)
        # weight_word_string = [int("".join([str(s) for s in weightmem_decoded[i][j:j+32]]),2) for j in range(0, 96, 32)]
        weight_int = [int(jjj, 2) for jjj in ["".join([str(jj) for jj in weightmem_decoded[i][j:j + 2]]) for j in range(0, 96, 2)]][::-1]
        weight_word_string = [int(jjj, 2) for jjj in ["".join(["{0:02b}".format(j) for j in weight_int])[jj:jj + 32] for jj in range(0, 96, 32)][::-1]]
        #f_weightmem_writes_intf.write("%d,%d,%08x,%08x,%08x\n" % (weightmemory_addr,weightmemory_bank, *weight_word_string))
        weightmem_writes_str += str('0x%08x,' % (weightmemory_addr*2**2 + weightmemory_bank*2**10 + WEIGHTMEM_START_ADDR)) # Print Address SCHEREMO: CHECK THIS
        for i in weight_word_string:
            weightmem_writes_str += str('0x%08x,' %  i) # Print Address
        weightmem_writes_str += str('\n') # Print Address

    f_weightmem_writes_c.write(weightmem_writes_str[:-2])
    #f_weightmem_writes_intf.close()
    f_weightmem_writes_c.write('};\n')
    f_weightmem_writes_c.write('#endif')
    f_weightmem_writes_c.close()
    # plt.matshow(weightmem_show)
    # plt.xticks([])
    # plt.yticks([])
    # plt.xlabel('Address Depth')
    # plt.ylabel('Banks')
    # plt.show()

    print("Generating thresholds stimuli file...")
    f_thresh_writes_c = open(os.path.join(out_directory, 'thresholds_intf.h'), 'w')
    f_thresh_writes_c.write('#ifndef __THRESHS_INCLUDE_GUARD\n')
    f_thresh_writes_c.write('#define __THRESHS_INCLUDE_GUARD\n\n')
    f_thresh_writes_c.write('uint32_t cutieThreshsLen[] = {%s};\n' % ', '.join([str(i) for i in layer_no]))
    f_thresh_writes_c.write('int16_t cutieThreshs[] PI_L2 = {\n')

    thresh_writes_str = ''

    for i in range(memwrites[-1]['thresh_writes']):
        if i >= memwrites[current_thresh_write_layer]['thresh_writes']:
            current_thresh_write_layer += 1
            thresh_addr = 0
        ocu_thresholds_save_enable = np.zeros(no, dtype=int)
        ocu_thresholds_save_enable[thresh_addr] = 1
        if (current_thresh_write_layer < num_cnn_layers):
            # ocu_thresh_pos = thresholds[thresh_addr,1]
            # ocu_thresh_neg = thresholds[thresh_addr,0]
            ocu_thresh_pos = net.cnn_thresh[current_thresh_write_layer].hi[thresh_addr]
            ocu_thresh_neg = net.cnn_thresh[current_thresh_write_layer].lo[thresh_addr]
        elif current_thresh_write_layer < num_layers - num_dense_layers:
            ocu_thresh_pos = net.tcn_thresh[current_thresh_write_layer - num_cnn_layers].hi[thresh_addr]
            ocu_thresh_neg = net.tcn_thresh[current_thresh_write_layer - num_cnn_layers].lo[thresh_addr]
        else:
            ocu_thresh_pos = 0
            ocu_thresh_neg = 0

        try:
            assert ocu_thresh_pos >= ocu_thresh_neg-1
        except AssertionError:
            assert False, "Got impossible threshold values!"

        thresh_addr += 1


        thresh_writes_str += str('0x{:04X},'.format(int(ocu_thresh_pos) & (2**16-1)))
        thresh_writes_str += str('0x{:04X},\n'.format(int(ocu_thresh_neg) & (2**16-1)))

    f_thresh_writes_c.write(thresh_writes_str[:-2])

    #f_thresh_intf.close()

    f_thresh_writes_c.write('};\n')
    f_thresh_writes_c.write('#endif')
    f_thresh_writes_c.close()

    print("Generating activation and result stimuli file...")

    f_actmem_writes_c = open(os.path.join(out_directory, 'activations_intf.h'), 'w')
    f_actmem_writes_c.write('#ifndef __ACTIVATIONS_INCLUDE_GUARD\n')
    f_actmem_writes_c.write('#define __ACTIVATIONS_INCLUDE_GUARD\n\n')
    f_actmem_writes_c.write('uint32_t cutieNumExecs = %d;\n' % num_execs)
    f_actmem_writes_c.write('uint32_t cutieActsLen = %d;\n' % (rounded_ni[0] // (ni // weight_stagger) * input_imagewidth * input_imageheight))
    #f_actmem_writes_c.write('uint32_t cutieActsLen = %d;\n' %  (input_imagewidth * input_imageheight))
    f_actmem_writes_c.write('int32_t cutieActs[] PI_L2 = {\n')

    f_responses_c = open(os.path.join(out_directory, 'responses_intf.h'), 'w')
    f_responses_c.write('#ifndef __RESPONSES_INCLUDE_GUARD\n')
    f_responses_c.write('#define __RESPONSES_INCLUDE_GUARD\n\n')
    f_responses_c.write('bool cutieUseFPoutput = {};\n'.format('false' if num_dense_layers==0 else 'true'))
    if num_dense_layers == 0:
        f_responses_c.write('uint32_t cutieResponsesLen = %d;\n' % (rounded_no[-1] // (no // weight_stagger) * torch.prod(torch.tensor(outshapes[-1][2:])).item()))
        #f_responses_c.write('uint32_t cutieResponsesLen = %d;\n' % (torch.prod(torch.tensor(outshapes[-1][2:])).item()))
    else:
        f_responses_c.write('uint32_t cutieResponsesLen = %d;\n' % (n_classes))
    f_responses_c.write('int32_t cutieResponses[] PI_L2 = {\n')


    activations_write_str = ''
    responses_write_str = ''
    for i in range(num_execs):
        # new_image, new_image_padded = make_random_image(input_imagewidth, input_imageheight, layer_ni[0], rounded_ni[0])

        # result, _ = net(new_image)
        # image = acts_f['arr_{}'.format(n_layers)]
        image = np.expand_dims(all_inputs[i], axis=0)

        encoded_image, decoded_image = translate_image_to_actmem(image)
        for addr, (enc_word, dec_word) in enumerate(zip(encoded_image, decoded_image)):
            act_int = [int(jjj, 2) for jjj in ["".join([str(jj) for jj in dec_word[j:j + 2]]) for j in range(0, 96, 2)]][::-1]
            act_word_string = [int(jjj, 2) for jjj in ["".join(["{0:02b}".format(j) for j in act_int])[jj:jj + 32] for jj in range(0, 96, 32)][::-1]]

            activations_write_str += str('0x%08x,' % (addr*2**2 + ACTMEM_START_ADDR)) # Print Address SCHEREMO: CHECK THIS
            for i in act_word_string:
                activations_write_str += str('0x%08x,' %  i) # Print Address
            activations_write_str += str('\n') # Print Address
        if num_dense_layers == 0:
            encoded_result, decoded_result = translate_image_to_actmem(all_results[i])
            for addr, (enc_word, dec_word) in enumerate(zip(encoded_result, decoded_result)):
                act_int = [int(jjj, 2) for jjj in ["".join([str(jj) for jj in dec_word[j:j + 2]]) for j in range(0, 96, 2)]][::-1]
                act_word_string = [int(jjj, 2) for jjj in ["".join(["{0:02b}".format(j) for j in act_int])[jj:jj + 32] for jj in range(0, 96, 32)][::-1]]

                responses_write_str += str('0x%08x,' % (ACTMEM_START_ADDR + addr*2**2)) # Print Address SCHEREMO: CHECK THIS
                for i in act_word_string:
                    responses_write_str += str('0x%08x,' %  i) # Print Address
                responses_write_str += str('\n') # Print Address
        else:
            for data in all_results[i].squeeze():
                if data < 0:
                    responses_write_str += "0x%s,\n" % (hex((int(data.item()) + (1 << 32)) % (1 << 32))[2:])
                else:
                    responses_write_str += "0x%08X,\n" % (int(data.item()))



    f_responses_c.write(responses_write_str[:-2])
    f_actmem_writes_c.write(activations_write_str[:-2])

    f_actmem_writes_c.write('};\n')
    f_actmem_writes_c.write('#endif')
    f_actmem_writes_c.close()

    f_responses_c.write('};\n')
    f_responses_c.write('#endif')
    f_responses_c.close()

    filepath = Path(__file__).parent.resolve()
    for fn in glob.glob(str(filepath.joinpath('kraken_c_proj'))+'/*'):
        shutil.copy2(fn, out_directory)
