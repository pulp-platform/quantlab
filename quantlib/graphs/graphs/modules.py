import torch
import torch.nn as nn


__all__ = [
    'ViewFlattenNd',
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
