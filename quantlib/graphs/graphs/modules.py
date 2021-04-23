import torch
import torch.nn as nn


__all__ = [
    'ViewFlattenNd',
    'ShiftAndClip',
    'FloorAndClip',
    'HelperInput',
    'HelperOutput',
    'HelperPrecisionTunnel',
    'HelperInputPrecisionTunnel',
    'HelperOutputPrecisionTunnel',
]


class ViewFlattenNd(nn.Module):

    def __init__(self):
        super(ViewFlattenNd, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ShiftAndClip(nn.Module):

    def __init__(self, n_bits=8, shift=17, signed=True, only_positive=False):
        super(ShiftAndClip, self).__init__()
        self._n_bits = n_bits
        self.shift = shift
        self.min = -2**(self._n_bits - 1) if signed else 0
        self.min = 0 if only_positive else self.min
        self.max = 2**(self._n_bits - 1) - 1 if signed else 2**self._n_bits - 1

    def forward(self, x):
        x //= 2**self.shift
        x = torch.clamp(x, self.min, self.max)
        return x


class FloorAndClip(nn.Module):

    def __init__(self, n_bits=8, signed=True, only_positive=False):
        super(FloorAndClip, self).__init__()
        self._n_bits = n_bits
        self.min = -2**(self._n_bits - 1) if signed else 0
        self.min = 0 if only_positive else self.min
        self.max = 2**(self._n_bits - 1) - 1 if signed else 2**self._n_bits - 1

    def forward(self, x):
        x = torch.floor(x)
        x = torch.clamp(x, self.min, self.max)
        return x


class HelperInput(nn.Identity):

    def __init__(self):
        super(HelperInput, self).__init__()


class HelperOutput(nn.Identity):

    def __init__(self):
        super(HelperOutput, self).__init__()


class HelperPrecisionTunnel(nn.Module):

    def __init__(self, eps_in, eps_out):
        super(HelperPrecisionTunnel, self).__init__()
        self.eps_in = eps_in
        self.eps_out = eps_out

    def forward(self, x):
        x = torch.div(x, self.eps_in)
        x = torch.mul(x, self.eps_out)
        return x


class HelperInputPrecisionTunnel(HelperPrecisionTunnel):

    def __init__(self, eps_in):
        super(HelperInputPrecisionTunnel, self).__init__(eps_in, 1.0)


class HelperOutputPrecisionTunnel(HelperPrecisionTunnel):

    def __init__(self, eps_out):
        super(HelperOutputPrecisionTunnel, self).__init__(1.0, eps_out)
