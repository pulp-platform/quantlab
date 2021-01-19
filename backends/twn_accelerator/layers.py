import numpy as np
import torch
import torch.nn as nn
import os
from typing import Union
from pathlib import Path

from quantlab.graphs.analyse import Node
from .quantops import STEActivationInteger
from .weights import export_tw, import_tw
from .gammabeta import export_gamma, import_gamma, export_beta, import_beta



def node_is_module(n : Node, m):

    if not isinstance(m, list):
        assert isinstance(m, type)
        m = [m]
    return any(isinstance(n.module, t) for t in m)


def layer_has_modules(l : Union[list, Node], m : list):
    if not isinstance(l, list):
        assert isinstance(l, Node), "Input to layer_has_modules must be Node or list of Nodes, not {}".format(l.__class__.__name__)
        l = [l]
    return any(node_is_module(n, m) for n in l)


def fold_h2d_layer(export_dir, w, eps, mu, sigma, gamma, beta, n_out, m_out, input_type='float'):

    def torch2numpy64(x):
        return x.detach().numpy().astype(np.float64)

    w     = torch2numpy64(w)
    mu    = torch2numpy64(mu)
    sigma = torch2numpy64(sigma)
    gamma = torch2numpy64(gamma)
    beta  = torch2numpy64(beta)
    m_out = torch2numpy64(m_out)

    sigma = np.sqrt(sigma + eps)  # sigma is the variance; at inference time, BatchNorm2d uses standard deviation:
                                  # https://pytorch.org/docs/stable/nn.html#batchnorm2d

    C_out, C_in, K1, K2 = w.shape

    w_temp = w

    if input_type == 'float':
        w_bias = np.zeros(C_out,)
    else:
        from problems.ImageNet.VGG.preprocess import _ImageNet
        mean = np.array(_ImageNet['Normalize']['mean']) * 255.
        std = np.array(_ImageNet['Normalize']['std']) * 255.
        w_temp = w_temp.transpose(0, 2, 3, 1)
        w_temp = w_temp / std
        w_bias = (w_temp * mean) / std
        w_bias = w_bias.transpose(0, 3, 1, 2).reshape(C_out, -1).sum(axis=1)
        if input_type == 'int8':
            w_bias += 128 * w_temp.reshape(C_out, -1).sum(axis=1)
        w_temp = w_temp.transpose(0, 3, 1, 2)

    ex_out = (2 * m_out) / (n_out - 1)

    w_temp = w_temp.transpose(1, 2, 3, 0)
    w_temp = (w_temp * gamma) / (ex_out * sigma)
    weight = w_temp.transpose(3, 0, 1, 2)

    weight = weight.transpose(0, 2, 3, 1)  # C_in is last dimension (design choice)
    with open(os.path.join(export_dir, 'weight'), 'wb') as fp:
        fp.write(weight.flatten().astype(np.float32))

    with open(os.path.join(export_dir, 'weight'), 'rb') as fp:
        buffer = np.frombuffer(fp.read(), dtype=np.float32)
    weight = buffer.reshape(weight.shape).astype(np.float64)  # set again as collection of 3D tensors
    weight = weight.transpose(0, 3, 1, 2)  # restore C_in in second position (to allow software simulation)

    # bias = (n_out - 1) * (((-mu * gamma) / (2 * m_out * sigma)) + (beta / (2 * m_out)) + 0.5)# + 0.5 using the `round` functional, not `floor`
    bias = ((((- w_bias - mu) * gamma) / sigma) + beta) / ex_out + 0.5

    with open(os.path.join(export_dir, 'bias'), 'wb') as fp:
        fp.write(bias.astype(np.float32))

    with open(os.path.join(export_dir, 'bias'), 'rb') as fp:
        buffer = np.frombuffer(fp.read(), dtype=np.float32)
    bias = buffer.astype(np.float64)

    def numpy2torch64(x):
        return torch.from_numpy(x.astype(np.float64))

    return numpy2torch64(weight), numpy2torch64(bias)


def fold_d2d_layer(export_dir, n_in, m_in, w, eps, mu, sigma, gamma, beta, n_out, m_out, params):

    def torch2numpy64(x):
        return x.detach().numpy().astype(np.float64)

    m_in  = torch2numpy64(m_in)
    w     = torch2numpy64(w)
    mu    = torch2numpy64(mu)
    sigma = torch2numpy64(sigma)
    gamma = torch2numpy64(gamma)
    beta  = torch2numpy64(beta)
    m_out = torch2numpy64(m_out)

    sigma = np.sqrt(sigma + eps)

    # compensate for negative gammas
    flip = np.sign(gamma)
    w_temp = w.transpose(1, 2, 3, 0)
    w_temp *= flip
    weight = w_temp.transpose(3, 0, 1, 2)

    ex_in = (2 * m_in) / (n_in - 1)
    ex_out = (2 * m_out) / (n_out - 1)

    gamma_t = (ex_in * gamma) / (ex_out * sigma)
    gamma_t *= flip
    gamma_t = gamma_t.reshape(-1, 1, 1, 1)

    # w_sum = w.reshape((w.shape[0], -1)).sum(axis=1)
    # beta_t = (n_out - 1) * (((((-m_in * w_sum) - mu) * gamma / sigma) + beta) / (2 * m_out) + 0.5)# + 0.5 using the `round` functional, not `floor`
    beta_t = (((-mu * gamma) / sigma) + beta) / ex_out + 0.5

    export_tw(weight, 'weight', export_dir=export_dir, T_in=params.blk_size, T_out=params.blk_size)
    weight = import_tw(weight, 'weight', export_dir=export_dir, T_in=params.blk_size, T_out=params.blk_size)

    export_gamma(gamma_t, 'gamma', params=params, export_dir=export_dir, int_bits=10, frac_bits=17)
    gamma_t = import_gamma(gamma_t, 'gamma', params=params, export_dir=export_dir)
    gamma_t = gamma_t.reshape(-1, 1, 1, 1)


    export_beta(beta_t, 'beta', params=params, export_dir=export_dir, int_bits=8, frac_bits=17, true_frac_bits=0)
    beta_t = import_beta(beta_t, 'beta', params=params, export_dir=export_dir, int_bits=8, frac_bits=17, true_frac_bits=0)

    def numpy2torch64(x):
        return torch.from_numpy(x.astype(np.float64))

    return numpy2torch64(weight), numpy2torch64(gamma_t), numpy2torch64(beta_t)


def fold_d2h_layer(export_dir, n_in, m_in, w, eps, mu, sigma, gamma, beta, params):

    def torch2numpy64(x):
        return x.detach().numpy().astype(np.float64)

    m_in  = torch2numpy64(m_in)
    w     = torch2numpy64(w)
    mu    = torch2numpy64(mu)
    sigma = torch2numpy64(sigma)
    gamma = torch2numpy64(gamma)
    beta  = torch2numpy64(beta)

    sigma = np.sqrt(sigma + eps)

    # compensate for negative gammas
    flip = np.sign(gamma)
    w_temp = w.transpose(1, 2, 3, 0)
    w_temp *= flip
    weight = w_temp.transpose(3, 0, 1, 2)

    # quantum
    ex_in = (2 * m_in) / (n_in - 1)

    #whoa there hold up!!! this ain't right.
    #gamma_t = (ex_in * gamma) / sigma
    # as input and output quanta are assumed to be the same, gamma is not scaled.
    # TODO change this for the case with a different "final"/output quantization
    gamma_t = gamma / sigma
    gamma_t *= flip
    gamma_t = gamma_t.reshape(-1, 1, 1, 1)

    # w_sum = w.reshape((w.shape[0], -1)).sum(axis=1)
    # beta_t = (n_out - 1) * (((((-m_in * w_sum) - mu) * gamma / sigma) + beta) / (2 * m_out) + 0.5)# + 0.5 using the `round` functional, not `floor`
    # beta_t = ((-mu * gamma) / sigma) + beta
    # as we are still in the quantized representation, beta does need to be scaled
    beta_t = (((-mu * gamma) / sigma) + beta) / ex_in + 0.5

    export_tw(weight, 'weight', export_dir=export_dir, T_in=params.blk_size, T_out=params.blk_size)
    # weight = import_tw(weight, 'weight', export_dir=export_dir, T_in=params.blk_size, T_out=params.blk_size)

    # This needs changing/clarification:
    # do we want to fold this BN into the next layer? if so, that needs to be
    # done offline. Otherwise, export it regularly for the accelerator (what is done now in this
    # hack version)
    # export+import gammas
    #with open(os.path.join(export_dir, 'gamma'), 'wb') as fp:
    #    fp.write(gamma_t.astype(np.float32))
    #
    #with open(os.path.join(export_dir, 'gamma'), 'rb') as fp:
    #    buffer = np.frombuffer(fp.read(), dtype=np.float32)
    #gamma_t = buffer.astype(np.float64)
    #gamma_t = gamma_t.reshape(-1, 1, 1, 1)
    #
    ## export+import betas
    #
    ##with open(os.path.join(export_dir, 'beta'), 'wb') as fp:
    ##    fp.write(beta_t.astype(np.float32))
    #
    ##with open(os.path.join(export_dir, 'beta'), 'rb') as fp:
    ##    buffer = np.frombuffer(fp.read(), dtype=np.float32)
    #beta_t = buffer.astype(np.float64)
    # FOR NOW JUST EXPORT AS IN d2d
    export_gamma(gamma_t, 'gamma', params=params, export_dir=export_dir, int_bits=10, frac_bits=17)
    #gamma_t = import_gamma(gamma_t, 'gamma', params=params, export_dir=export_dir)
    #gamma_t = gamma_t.reshape(-1, 1, 1, 1)

    export_beta(beta_t, 'beta', params=params, export_dir=export_dir, int_bits=8, frac_bits=17, true_frac_bits=0)
    #beta_t = import_beta(beta_t, 'beta', params=params, export_dir=export_dir, int_bits=8, frac_bits=17, true_frac_bits=0)

    def numpy2torch64(x):
        return torch.from_numpy(x.astype(np.float64))

    return numpy2torch64(weight), numpy2torch64(gamma_t), numpy2torch64(beta_t)


def convert_h2d(layer, export_dir, input_type='float'):

    # parse layer into nodes
    conv_nodes   = [n[1] for n in layer if n[1].__class__.__name__ == 'Conv2d']
    bn_nodes     = [n[1] for n in layer if n[1].__class__.__name__ == 'BatchNorm2d']
    relu_nodes   = [n[1] for n in layer if n[1].__class__.__name__ == 'ReLU']
    ste_nodes    = [n[1] for n in layer if n[1].__class__.__name__ == 'STEActivation']

    # fold parameters
    conv = conv_nodes[0]
    bn   = bn_nodes[0]
    ste  = ste_nodes[0]
    weight, bias = fold_h2d_layer(export_dir,
                                  conv.weight,
                                  bn.eps, bn.running_mean, bn.running_var, bn.weight, bn.bias,
                                  ste.num_levels, ste.abs_max_value,
                                  input_type=input_type)

    # create SW-emulated layer
    new_conv = nn.Conv2d(in_channels=conv.in_channels, out_channels=conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding, bias=True)
    new_conv.weight.data = weight
    new_conv.bias.data = bias
    new_relu = nn.ReLU()
    # new_ste = qa.ste.STEActivationInteger(num_levels=ste.num_levels, zero_level=((ste.num_levels - 1) / 2))
    new_ste = STEActivationInteger(num_levels=ste.num_levels, is_input_integer=False, clamp_min_to_zero=False)

    nodes = [new_conv, new_relu, new_ste]

    return nodes


def convert_d2d(layer, export_dir, params):

    # parse layer into nodes
    ste_nodes    = [n[1] for n in layer if n[1].__class__.__name__ == 'STEActivation']
    conv_nodes   = [n[1] for n in layer if n[1].__class__.__name__ == 'INQConv2d']
    bn_nodes     = [n[1] for n in layer if n[1].__class__.__name__ == 'BatchNorm2d']
    relu_nodes   = [n[1] for n in layer if n[1].__class__.__name__ == 'ReLU']
    maxpool_node = [n[1] for n in layer if n[1].__class__.__name__ == 'MaxPool2d']
    assert len(ste_nodes)    == 2
    assert len(conv_nodes)   == 1
    assert len(bn_nodes)     == 1
    assert len(relu_nodes)   == 1
    assert len(maxpool_node) <= 1

    # fold parameters
    ste_in  = ste_nodes[0]
    conv    = conv_nodes[0]
    bn      = bn_nodes[0]
    ste_out = ste_nodes[1]
    weight, gamma_t, beta_t = fold_d2d_layer(export_dir,
                                             ste_in.num_levels, ste_in.abs_max_value,
                                             conv.weight_frozen,
                                             bn.eps, bn.running_mean, bn.running_var, bn.weight, bn.bias,
                                             ste_out.num_levels, ste_out.abs_max_value, params)

    # create SW-emulated layer
    new_conv_tw = nn.Conv2d(in_channels=conv.in_channels, out_channels=conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding, bias=conv.bias)
    new_conv_tw.weight.data = weight
    assert new_conv_tw.bias is None
    new_conv_fp = nn.Conv2d(in_channels=conv.out_channels, out_channels=conv.out_channels, kernel_size=1, stride=1, padding=0, groups=conv.out_channels, bias=True)
    new_conv_fp.weight.data = gamma_t
    new_conv_fp.bias.data = beta_t
    # new_ste = qa.ste.STEActivationInteger(num_levels=ste_out.num_levels, zero_level=((ste_out.num_levels - 1) / 2))
    new_ste = STEActivationInteger(num_levels=ste_out.num_levels, is_input_integer=True)

    nodes = [new_conv_tw, new_conv_fp, new_ste]
    if maxpool_node:
        nodes += [nn.MaxPool2d(kernel_size=maxpool_node[0].kernel_size, stride=maxpool_node[0].kernel_size)]

    return nodes


def convert_d2h(layer, export_dir, params):

    # parse layer into nodes
    ste_nodes    = [n[1] for n in layer if n[1].__class__.__name__ == 'STEActivation']
    conv_nodes   = [n[1] for n in layer if n[1].__class__.__name__ == 'INQConv2d']
    bn_nodes     = [n[1] for n in layer if n[1].__class__.__name__ == 'BatchNorm2d']
    relu_nodes   = [n[1] for n in layer if n[1].__class__.__name__ == 'ReLU']
    maxpool_node = [n[1] for n in layer if n[1].__class__.__name__ == 'MaxPool2d']
    # for now, assume the output quantization is the same as the input quantization
    # TODO: change this after training a new VGG
    assert len(ste_nodes)    == 1
    assert len(conv_nodes)   == 1
    assert len(bn_nodes)     == 1
    assert len(relu_nodes)   == 1
    assert len(maxpool_node) <= 1

    # fold parameters
    conv = conv_nodes[0]
    bn   = bn_nodes[0]
    ste  = ste_nodes[0]
    weight, gamma_t, beta_t = fold_d2h_layer(export_dir,
                                             ste.num_levels, ste.abs_max_value,
                                             conv.weight_frozen,
                                             bn.eps, bn.running_mean, bn.running_var, bn.weight, bn.bias, params)

    # create SW-emulated layer
    new_conv_tw = nn.Conv2d(in_channels=conv.in_channels, out_channels=conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding, bias=conv.bias)
    new_conv_tw.weight.data = weight
    assert new_conv_tw.bias is None
    new_conv_fp = nn.Conv2d(in_channels=conv.out_channels, out_channels=conv.out_channels, kernel_size=1, stride=1, padding=0, groups=conv.out_channels, bias=True)
    new_conv_fp.weight.data = gamma_t
    new_conv_fp.bias.data = beta_t
    new_relu = nn.ReLU(inplace=True)

    nodes = [new_conv_tw, new_conv_fp, new_relu]
    if maxpool_node:
        nodes += [nn.MaxPool2d(kernel_size=maxpool_node[0].kernel_size, stride=maxpool_node[0].kernel_size)]

    return nodes


def convert_h2h(layer, export_dir):
    # parse layer into nodes
    linear_nodes = [n[1] for n in layer if n[1].__class__.__name__ == 'Linear']
    relu_nodes   = [n[1] for n in layer if n[1].__class__.__name__ == 'ReLU']
    assert len(linear_nodes) == 1
    assert len(relu_nodes)   <= 1

    # create SW-emulated layer
    lin = linear_nodes[0]
    wt = lin.weight.data.cpu().clone().detach()
    bias = False if lin.bias is None else lin.bias.data.cpu().clone().detach()
    new_linear = nn.Linear(in_features=lin.in_features, out_features=lin.out_features, bias=bias is not False)
    new_linear.weight.data = lin.weight
    new_linear.bias.data = lin.bias

    nodes = [new_linear]
    if len(relu_nodes) > 1:
        nodes += [nn.ReLU(inplace=True)]

    # just dump the FC layer weights out.
    Path(export_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(export_dir, "weight"), "wb") as fh:
        for el in wt.float().numpy().flatten():
            fh.write(el)
    if bias is not False:
        with open(os.path.join(export_dir, "bias"), "wb") as fh:
            for el in bias.float().numpy().flatten():
                fh.write(el)

    return nodes