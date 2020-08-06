import numpy as np
import torch
import torch.nn as nn

from .quantops import STEActivationInteger
from .weights import export_tw, import_tw
from .gammabeta import export_gamma, import_gamma, export_beta, import_beta


def fold_h2d_layer(export_dir, w, eps, mu, sigma, gamma, beta, n_out, m_out):

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

    ex_out = (2 * m_out) / (n_out - 1)

    w_temp = w.transpose(1, 2, 3, 0)
    w_temp = (w_temp * gamma) / (ex_out * sigma)
    weight = w_temp.transpose(3, 0, 1, 2)

    # bias = (n_out - 1) * (((-mu * gamma) / (2 * m_out * sigma)) + (beta / (2 * m_out)) + 0.5)# + 0.5 using the `round` functional, not `floor`
    bias = (((-mu * gamma) / sigma) + beta) / ex_out + 0.5

    def numpy2torch64(x):
        return torch.from_numpy(x.astype(np.float64))

    return numpy2torch64(weight), numpy2torch64(bias)


def fold_d2d_layer(export_dir, n_in, m_in, w, eps, mu, sigma, gamma, beta, n_out, m_out):

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

    export_tw(weight, 'weight', export_dir=export_dir)
    weight = import_tw(weight, 'weight', export_dir=export_dir)

    export_gamma(gamma_t, 'gamma', export_dir=export_dir, int_bits=10, frac_bits=17)
    gamma_t = import_gamma(gamma_t, 'gamma', export_dir=export_dir)
    gamma_t = gamma_t.reshape(-1, 1, 1, 1)

    # export_beta(beta_t, 'beta', export_dir=export_dir, int_bits=8, frac_bits=17)
    # beta_t = import_beta(beta_t, 'beta', export_dir=export_dir)

    quantum = 2**(-17)
    beta_t /= quantum
    beta_t = beta_t.astype(np.int64)

    def numpy2torch64(x):
        return torch.from_numpy(x.astype(np.float64))

    return numpy2torch64(weight), numpy2torch64(gamma_t), numpy2torch64(beta_t)


def fold_d2h_layer(export_dir, n_in, m_in, w, eps, mu, sigma, gamma, beta):

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

    ex_in = (2 * m_in) / (n_in - 1)

    gamma_t = (ex_in * gamma) / sigma
    gamma_t *= flip
    gamma_t = gamma_t.reshape(-1, 1, 1, 1)

    # w_sum = w.reshape((w.shape[0], -1)).sum(axis=1)
    # beta_t = (n_out - 1) * (((((-m_in * w_sum) - mu) * gamma / sigma) + beta) / (2 * m_out) + 0.5)# + 0.5 using the `round` functional, not `floor`
    beta_t = ((-mu * gamma) / sigma) + beta

    export_tw(weight, 'weight', export_dir=export_dir)
    weight = import_tw(weight, 'weight', export_dir=export_dir)

    # export_gamma(gamma_t, 'gamma', export_dir=export_dir, int_bits=10, frac_bits=17)
    # gamma_t = import_gamma(gamma_t, 'gamma', export_dir=export_dir)
    # gamma_t = gamma_t.reshape(-1, 1, 1, 1)

    # export_beta(beta_t, 'beta', export_dir=export_dir, int_bits=8, frac_bits=17)
    # beta_t = import_beta(beta_t, 'beta', export_dir=export_dir)

    def numpy2torch64(x):
        return torch.from_numpy(x.astype(np.float64))

    return numpy2torch64(weight), numpy2torch64(gamma_t), numpy2torch64(beta_t)


def convert_h2d(layer, export_dir):

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
                                  ste.num_levels, ste.abs_max_value)

    # create SW-emulated layer
    new_conv = nn.Conv2d(in_channels=conv.in_channels, out_channels=conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding, bias=True)
    new_conv.weight.data = weight
    new_conv.bias.data = bias
    # new_ste = qa.ste.STEActivationInteger(num_levels=ste.num_levels, zero_level=((ste.num_levels - 1) / 2))
    new_ste = STEActivationInteger(num_levels=ste.num_levels, is_input_integer=False)

    nodes = [new_conv, new_ste]

    return nn.Sequential(*nodes)


def convert_d2d(layer, export_dir):

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
                                             ste_out.num_levels, ste_out.abs_max_value)

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

    return nn.Sequential(*nodes)


def convert_d2h(layer, export_dir):

    # parse layer into nodes
    ste_nodes    = [n[1] for n in layer if n[1].__class__.__name__ == 'STEActivation']
    conv_nodes   = [n[1] for n in layer if n[1].__class__.__name__ == 'INQConv2d']
    bn_nodes     = [n[1] for n in layer if n[1].__class__.__name__ == 'BatchNorm2d']
    relu_nodes   = [n[1] for n in layer if n[1].__class__.__name__ == 'ReLU']
    maxpool_node = [n[1] for n in layer if n[1].__class__.__name__ == 'MaxPool2d']
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
                                             bn.eps, bn.running_mean, bn.running_var, bn.weight, bn.bias)

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

    return nn.Sequential(*nodes)


def convert_h2h(layer, export_dir):

    # parse layer into nodes
    linear_nodes = [n[1] for n in layer if n[1].__class__.__name__ == 'Linear']
    relu_nodes   = [n[1] for n in layer if n[1].__class__.__name__ == 'ReLU']
    assert len(linear_nodes) == 1
    assert len(relu_nodes)   <= 1

    # create SW-emulated layer
    lin = linear_nodes[0]
    new_linear = nn.Linear(in_features=lin.in_features, out_features=lin.out_features, bias=False if lin.bias is None else True)
    new_linear.weight.data = lin.weight
    new_linear.bias.data = lin.bias

    nodes = [new_linear]
    if len(relu_nodes) > 1:
        nodes += [nn.ReLU(inplace=True)]

    return nn.Sequential(*nodes)
