import math
# from scipy.stats import norm, uniform
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple


__all__ = [
    'ANAActivation',
    'ANALinear',
    'ANAConv1d',
    'ANAConv2d',
    'ANAConv3d',
]


class UniformHeavisideProcess(torch.autograd.Function):
    """A stochastic process composed by step functions.

    This class defines a stochastic process whose elementary events are step
    functions with fixed quantization levels (codominion) and uniform noise on
    the jumps positions.
    """
    @staticmethod
    def forward(ctx, x, t, q, s, training):
        ctx.save_for_backward(x, t, q, s)
        t_shape = [t.numel()] + [1 for _ in range(x.dim())]  # dimensions with size 1 enable broadcasting
        x_minus_t = x - t.reshape(t_shape)
        if training and s[0] != 0.:
            s_inv_for = 1 / s[0]
            cdf = torch.clamp(0.5 * (x_minus_t * s_inv_for + 1), 0., 1.)
        else:
            cdf = (x_minus_t >= 0.).float()
        d = q[1:] - q[:-1]
        sigma_x = q[0] + torch.sum(d.reshape(t_shape) * cdf, 0)
        return sigma_x

    @staticmethod
    def backward(ctx, grad_incoming):
        x, t, q, s = ctx.saved_tensors
        t_shape = [t.numel()] + [1 for _ in range(x.dim())]  # dimensions with size 1 enable broadcasting
        x_minus_t = x - t.reshape(t_shape)
        if s[1] != 0.:
            s_inv_back = 1 / s[1]
            pdf = (torch.abs_(x_minus_t) <= s_inv_back[1]).float() * (0.5 * s_inv_back)
        else:
            pdf = torch.zeros_like(grad_incoming)
        d = q[1:] - q[:-1]
        local_jacobian = torch.sum(d.reshape(t_shape) * pdf, 0)
        grad_outgoing = grad_incoming * local_jacobian
        return grad_outgoing, None, None, None, None


class ANAActivation(nn.Module):
    """Quantize scores."""
    def __init__(self, process, thresholds, quant_levels):
        super(ANAActivation, self).__init__()
        self.process = process
        if self.process == 'uniform':
            self.activate = UniformHeavisideProcess.apply
        super(ANAActivation, self).register_parameter('thresholds',
                                                             nn.Parameter(torch.Tensor(thresholds),
                                                                          requires_grad=False))
        super(ANAActivation, self).register_parameter('quant_levels',
                                                             nn.Parameter(torch.Tensor(quant_levels),
                                                                          requires_grad=False))
        super(ANAActivation, self).register_parameter('stddev',
                                                             nn.Parameter(torch.Tensor(torch.ones(2)),
                                                                          requires_grad=False))

    def set_stddev(self, stddev):
        self.stddev.data = torch.Tensor(stddev).to(self.stddev)

    def forward(self, x):
        return self.activate(x, self.thresholds, self.quant_levels, self.stddev, self.training)


class ANALinear(nn.Module):
    """Affine transform with quantized parameters."""
    def __init__(self, process, thresholds, quant_levels, in_features, out_features, bias=True):
        super(ANALinear, self).__init__()
        # set stochastic properties
        self.process = process
        if self.process == 'uniform':
            self.activate_weight = UniformHeavisideProcess.apply
        super(ANALinear, self).register_parameter('thresholds',
                                                         nn.Parameter(torch.Tensor(thresholds),
                                                                      requires_grad=False))
        super(ANALinear, self).register_parameter('quant_levels',
                                                         nn.Parameter(torch.Tensor(quant_levels),
                                                                      requires_grad=False))
        super(ANALinear, self).register_parameter('stddev',
                                                         nn.Parameter(torch.Tensor(torch.ones(2)),
                                                                      requires_grad=False))
        # set linear layer properties
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

    def set_stddev(self, stddev):
        self.stddev.data = torch.Tensor(stddev).to(self.stddev)

    def forward(self, input):
        weight = self.activate_weight(self.weight, self.thresholds, self.quant_levels, self.stddev, self.training)
        return F.linear(input, weight, self.bias)


class _ANAConvNd(nn.Module):
    """Cross-correlation transform with quantized parameters."""
    def __init__(self, process, thresholds, quant_levels,
                 in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, bias):
        super(_ANAConvNd, self).__init__()
        # set stochastic properties
        self.process = process
        if self.process == 'uniform':
            self.activate_weight = UniformHeavisideProcess.apply
        super(_ANAConvNd, self).register_parameter('thresholds',
                                                          nn.Parameter(torch.Tensor(thresholds),
                                                                       requires_grad=False))
        super(_ANAConvNd, self).register_parameter('quant_levels',
                                                          nn.Parameter(torch.Tensor(quant_levels),
                                                                       requires_grad=False))
        super(_ANAConvNd, self).register_parameter('stddev',
                                                          nn.Parameter(torch.Tensor(torch.ones(2)),
                                                                       requires_grad=False))
        # set convolutional layer properties
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

    def set_stddev(self, stddev):
        self.stddev.data = torch.Tensor(stddev).to(self.stddev)


class ANAConv1d(_ANAConvNd):
    def __init__(self, process, thresholds, quant_levels,
                 in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _single(kernel_size)
        stride      = _single(stride)
        padding     = _single(padding)
        dilation    = _single(dilation)
        super(ANAConv1d, self).__init__(
              process, thresholds, quant_levels,
              in_channels, out_channels, kernel_size, stride, padding, dilation, False, _single(0), groups, bias)

    def forward(self, input):
        weight = self.activate_weight(self.weight, self.thresholds, self.quant_levels, self.stddev, self.training)
        return F.conv1d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ANAConv2d(_ANAConvNd):
    def __init__(self, process, thresholds, quant_levels,
                 in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride      = _pair(stride)
        padding     = _pair(padding)
        dilation    = _pair(dilation)
        super(ANAConv2d, self).__init__(
              process, thresholds, quant_levels,
              in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias)

    def forward(self, input):
        weight = self.activate_weight(self.weight, self.thresholds, self.quant_levels, self.stddev, self.training)
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ANAConv3d(_ANAConvNd):
    def __init__(self, process, thresholds, quant_levels,
                 in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _triple(kernel_size)
        stride      = _triple(stride)
        padding     = _triple(padding)
        dilation    = _triple(dilation)
        super(ANAConv3d, self).__init__(
              process, thresholds, quant_levels,
              in_channels, out_channels, kernel_size, stride, padding, dilation, False, _triple(0), groups, bias)

    def forward(self, input):
        weight = self.activate_weight(self.weight, self.thresholds, self.quant_levels, self.stddev, self.training)
        return F.conv3d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
