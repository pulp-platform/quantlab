import torch.nn as nn
from collections import OrderedDict

import quantlab.algorithms as algo
from .analyse import list_nodes


__all__ = [
    'add_after_conv2d_per_ch_affine',
    'add_before_linear_ste',
    'replace_linear_inq',
]


def _get_node(net, name):
    """Return a handle on a specified node in the network's graph.

    This can be used to extract parameters for builder methods of quantized nodes."""
    path_to_node = name.split('.', 1)
    if len(path_to_node) == 1:
        node = net._modules[path_to_node[0]]
    else:
        node = _get_node(net._modules[path_to_node[0]], path_to_node[1])
    return node


def _replace_node(net, name, new_node):
    """Replace a specified node in the network's graph with a quantized counterpart."""
    path_to_node = name.split('.', 1)
    if len(path_to_node) == 1:
        # node = net._modules[path_to_node[0]]
        net._modules[path_to_node[0]] = new_node
    else:
        _replace_node(net._modules[path_to_node[0]], path_to_node[1], new_node)
    return


def add_after_conv2d_per_ch_affine(net, nodes_set):

    # when I am not supposed to use this symbol in other places (i.e., outside of the scope of the function where it is
    # called), it is better to embed its definition inside the function's definition itself
    class Affine(nn.Conv2d):
        def __init__(self, n_channels):
            super(Affine, self).__init__(n_channels, n_channels, kernel_size=1, stride=1, padding=0, groups=n_channels, bias=True)
            self.weight.data.fill_(1.)
            self.bias.data.fill_(0.)

    for n, _ in nodes_set:
        m = _get_node(net, n)
        if m.__class__.__name__ == 'Conv2d':# and m.bias is not None:  # if it does not have bias, there is a BN layer afterwards
            m.bias = None
            node = Affine(m.out_channels)
            _replace_node(net, n, nn.Sequential(OrderedDict([('conv', m), ('affine', node)])))


def add_before_linear_ste(net, nodes_set, num_levels, quant_start_epoch=0):
    for n, _ in nodes_set:
        ste_node = algo.ste.STEActivation(num_levels=num_levels, quant_start_epoch=quant_start_epoch)
        m = _get_node(net, n)
        _replace_node(net, n, nn.Sequential(OrderedDict([('ste', ste_node), ('conv', m)])))


def replace_linear_inq(net, nodes_set, num_levels, quant_init_method=None, quant_strategy='magnitude'):
    """Replace nodes representing linear operations with INQ counterparts.

    Non-linear nodes are not a target for INQ, which was developed to train weight-only quantized networks."""
    for n, _ in nodes_set:
        m = _get_node(net, n)
        m_type = m.__class__.__name__
        inq_node = None
        if m_type == 'Linear':
            in_features = m.in_features
            out_features = m.out_features
            bias = m.bias
            inq_node = algo.inq.INQLinear(in_features, out_features, bias=bias,
                                          num_levels=num_levels, quant_init_method=quant_init_method, quant_strategy=quant_strategy)
        elif m_type.startswith('Conv'):
            in_channels = m.in_channels
            out_channels = m.out_channels
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            dilation = m.dilation
            groups = m.groups
            bias = m.bias
            if m_type == 'Conv1d':
                inq_node = algo.inq.INQConv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                                              num_levels=num_levels, quant_init_method=quant_init_method, quant_strategy=quant_strategy)
            if m_type == 'Conv2d':
                inq_node = algo.inq.INQConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                                              num_levels=num_levels, quant_init_method=quant_init_method, quant_strategy=quant_strategy)
            if m_type == 'Conv3d':
                raise NotImplementedError
        assert(inq_node is not None)
        _replace_node(net, n, inq_node)
    return list_nodes(net)
