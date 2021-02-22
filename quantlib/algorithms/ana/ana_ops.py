import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple

import math
from collections import namedtuple

from . import ana_lib


__all__ = [
    'ANAActivation',
    'ANALinear',
    'ANAConv1d',
    'ANAConv2d',
    'ANAConv3d',
]


Noise = namedtuple('Noise', ['mu', 'sigma'])


class ANAModule(nn.Module):

    def __init__(self, quantizer_spec, noise_type):
        super(ANAModule, self).__init__()
        ANAModule.setup_quantizer(self, quantizer_spec)
        ANAModule.setup_noise(self, noise_type)

    @staticmethod
    def setup_quantizer(anamod, quantizer_spec):
        """The quantizer is a stair function specified by:
        * number of bits;
        * unsigned vs. signed integer representation;
        * if signed, unbalanced or balanced wrt zero (i.e., the exceeding negative value is discarded);
        * quantum (i.e., the precision of the fixed-point representation).
        """

        # quantization levels
        quant_levels = torch.arange(0, 2 ** quantizer_spec['nbits']).to(dtype=torch.float32)
        if quantizer_spec['signed']:
            quant_levels = quant_levels - 2 ** (quantizer_spec['nbits'] - 1)
            if quantizer_spec['balanced']:
                quant_levels = quant_levels[1:]
        anamod.register_parameter('quant_levels', nn.Parameter(quant_levels, requires_grad=False))

        # thresholds
        thresholds = quant_levels[:-1] + .5
        anamod.register_parameter('thresholds', nn.Parameter(thresholds, requires_grad=False))

        # quantum
        eps = torch.Tensor([quantizer_spec['eps']])
        anamod.register_parameter('eps', nn.Parameter(eps, requires_grad=False))

    @staticmethod
    def setup_noise(anamod, noise_type):

        # noise type
        anamod.ana_op = getattr(ana_lib, 'ANA' + noise_type.capitalize()).apply

        # initialise forward noise parameters
        anamod.register_parameter('fmu', nn.Parameter(torch.zeros(1), requires_grad=False))
        anamod.register_parameter('fsigma', nn.Parameter(torch.ones(1), requires_grad=False))
        anamod.fnoise = Noise(mu=anamod.fmu, sigma=anamod.fsigma)

        # initialise backward noise parameters
        anamod.register_parameter('bmu', nn.Parameter(torch.zeros(1), requires_grad=False))
        anamod.register_parameter('bsigma', nn.Parameter(torch.ones(1), requires_grad=False))
        anamod.bnoise = Noise(mu=anamod.bmu, sigma=anamod.bsigma)

    def set_fnoise(self, mu, sigma):
        self.fnoise.mu.data = torch.Tensor([mu]).to(self.fnoise.mu)
        self.fnoise.sigma.data = torch.Tensor([sigma]).to(self.fnoise.sigma)

    def set_bnoise(self, mu, sigma):
        self.bnoise.mu.data = torch.Tensor([mu]).to(self.bnoise.mu)
        self.bnoise.sigma.data = torch.Tensor([sigma]).to(self.bnoise.sigma)


class ANAActivation(ANAModule):
    """Quantize scores."""
    def __init__(self, quantizer_spec, noise_type):
        super(ANAActivation, self).__init__(quantizer_spec, noise_type)

    def forward(self, x):

        x = x / self.eps if self.eps != 1.0 else x

        x_out = self.ana_op(x,
                            self.quant_levels, self.thresholds,
                            self.fnoise.mu, self.fnoise.sigma,
                            self.bnoise.mu, self.bnoise.sigma,
                            self.training)

        x_out = x_out * self.eps if self.eps != 1.0 else x_out

        return x_out


class ANALinear(ANAModule):
    """Affine transform with quantized parameters."""
    def __init__(self, quantizer_spec, noise_type,
                 in_features, out_features, bias=True):

        # set quantizer + ANA properties
        super(ANALinear, self).__init__(quantizer_spec, noise_type)
        assert self.eps == 1.0  # quantized weights should be "pure" integers, not fixed-point numbers

        # set linear layer parameters
        self.in_features  = in_features
        self.out_features = out_features
        self.weight       = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))

        # init weights near thresholds
        self.weight.data.random_(to=len(self.thresholds.data))
        self.weight.data = self.thresholds[self.weight.data.to(torch.long)]
        self.weight.data = torch.add(self.weight.data, torch.zeros_like(self.weight.data).uniform_(-stdv, stdv))

        # init biases
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):

        weight = self.ana_op(self.weight,
                             self.quant_levels, self.thresholds,
                             self.fnoise.mu, self.fnoise.sigma,
                             self.bnoise.mu, self.bnoise.sigma,
                             self.training)

        return F.linear(input, weight, self.bias)


class _ANAConvNd(ANAModule):
    """Cross-correlation transform with quantized parameters."""
    def __init__(self, quantizer_spec, noise_type,
                 in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, bias):

        # set quantizer + ANA properties
        super(_ANAConvNd, self).__init__(quantizer_spec, noise_type)
        assert self.eps == 1.0  # quantized weights should be "pure" integers, not fixed-point numbers

        # set convolutional layer parameters
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.kernel_size    = kernel_size
        self.stride         = stride
        self.padding        = padding
        self.dilation       = dilation
        self.transposed     = transposed
        self.output_padding = output_padding
        self.groups         = groups
        if transposed:
            self.weight = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):

        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)

        # init weights near thresholds
        self.weight.data.random_(to=len(self.thresholds.data))
        self.weight.data = self.thresholds[self.weight.data.to(torch.long)]
        self.weight.data = torch.add(self.weight.data, torch.zeros_like(self.weight.data).uniform_(-stdv, stdv))

        # init biases
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


class ANAConv1d(_ANAConvNd):
    def __init__(self, quantizer_spec, noise_type,
                 in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):

        kernel_size = _single(kernel_size)
        stride      = _single(stride)
        padding     = _single(padding)
        dilation    = _single(dilation)

        super(ANAConv1d, self).__init__(
            quantizer_spec, noise_type,
            in_channels, out_channels, kernel_size, stride, padding, dilation, False, _single(0), groups, bias
        )

    def forward(self, input):

        weight = self.ana_op(self.weight,
                             self.quant_levels, self.thresholds,
                             self.fnoise.mu, self.fnoise.sigma,
                             self.bnoise.mu, self.bnoise.sigma,
                             self.training)

        return F.conv1d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ANAConv2d(_ANAConvNd):
    def __init__(self, quantizer_spec, noise_type,
                 in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):

        kernel_size = _pair(kernel_size)
        stride      = _pair(stride)
        padding     = _pair(padding)
        dilation    = _pair(dilation)

        super(ANAConv2d, self).__init__(
            quantizer_spec, noise_type,
            in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias
        )

    def forward(self, input):

        weight = self.ana_op(self.weight,
                             self.quant_levels, self.thresholds,
                             self.fnoise.mu, self.fnoise.sigma,
                             self.bnoise.mu, self.bnoise.sigma,
                             self.training)

        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ANAConv3d(_ANAConvNd):
    def __init__(self, quantizer_spec, noise_type,
                 in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):

        kernel_size = _triple(kernel_size)
        stride      = _triple(stride)
        padding     = _triple(padding)
        dilation    = _triple(dilation)

        super(ANAConv3d, self).__init__(
            quantizer_spec, noise_type,
            in_channels, out_channels, kernel_size, stride, padding, dilation, False, _triple(0), groups, bias
        )

    def forward(self, input):

        weight = self.ana_op(self.weight,
                             self.quant_levels, self.thresholds,
                             self.fnoise.mu, self.fnoise.sigma,
                             self.bnoise.mu, self.bnoise.sigma,
                             self.training)

        return F.conv3d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
