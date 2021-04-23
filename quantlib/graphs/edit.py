import torch.nn as nn
from collections import OrderedDict
import importlib
import copy

import quantlib.algorithms as qa
from .analyse import list_nodes


__all__ = [
    'add_after_conv2d_per_ch_affine',
    'add_before_linear_ste',
    'replace_linear_inq',
]


def get_module(parent_module, target_name):
    """Return a handle on a specified module in the network's graph.

    For example, if the goal is to replace the target module with a quantized
    counterpart, the retrieved module can be used to extract the structural
    parameters that need to be passed to the constructor method of the
    quantized module (e.g., the kernel size for convolutional layers).
    """

    path_to_target = target_name.split('.', 1)

    if len(path_to_target) == 1:
        module = parent_module._modules[path_to_target[0]]
    else:
        module = get_module(parent_module._modules[path_to_target[0]], path_to_target[1])

    return module


def replace_module(parent_module, target_name, new_module):
    """Replace a specified module with a given counterpart.

    For example, this function can be used to replace full-precision PyTorch
    modules with quantized counterparts defined in `quantlib.algorithms`.
    """

    path_to_target = target_name.split('.', 1)

    if len(path_to_target) == 1:
        # node = net._modules[path_to_node[0]]
        parent_module._modules[path_to_target[0]] = new_module
    else:
        replace_module(parent_module._modules[path_to_target[0]], path_to_target[1], new_module)


def add_after_conv2d_per_ch_affine(net, nodes_set):

    # when I am not supposed to use this symbol in other places (i.e., outside of the scope of the function where it is
    # called), it is better to embed its definition inside the function's definition itself
    class Affine(nn.Conv2d):
        def __init__(self, n_channels):
            super(Affine, self).__init__(n_channels, n_channels, kernel_size=1, stride=1, padding=0, groups=n_channels, bias=True)
            self.weight.data.fill_(1.)
            self.bias.data.fill_(0.)

    for n, _ in nodes_set:
        m = get_module(net, n)
        if m.__class__.__name__ == 'Conv2d':# and m.bias is not None:  # if it does not have bias, there is a BN layer afterwards
            m.bias = None
            node = Affine(m.out_channels)
            replace_module(net, n, nn.Sequential(OrderedDict([('conv', m), ('affine', node)])))


def add_before_linear_ste(net, nodes_set, num_levels, quant_start_epoch=0):
    for n, _ in nodes_set:
        ste_node = qa.ste.STEActivation(num_levels=num_levels, quant_start_epoch=quant_start_epoch)
        m = get_module(net, n)
        replace_module(net, n, nn.Sequential(OrderedDict([('ste', ste_node), ('conv', m)])))


def replace_linear_inq(net, nodes_set, num_levels, quant_init_method=None, quant_strategy='magnitude'):
    """Replace nodes representing linear operations with INQ counterparts.

    Non-linear nodes are not a target for INQ, which was developed to train weight-only quantized networks."""
    for n, _ in nodes_set:
        m = get_module(net, n)
        m_type = m.__class__.__name__
        inq_node = None
        if m_type == 'Linear':
            in_features = m.in_features
            out_features = m.out_features
            bias = m.bias
            inq_node = qa.inq.INQLinear(in_features, out_features, bias=bias,
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
                inq_node = qa.inq.INQConv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                                            num_levels=num_levels, quant_init_method=quant_init_method, quant_strategy=quant_strategy)
            if m_type == 'Conv2d':
                inq_node = qa.inq.INQConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                                            num_levels=num_levels, quant_init_method=quant_init_method, quant_strategy=quant_strategy)
            if m_type == 'Conv3d':
                raise NotImplementedError
        assert(inq_node is not None)
        replace_module(net, n, inq_node)
    return list_nodes(net)


class Loader(object):

    def __init__(self, problem, topology, config):
        self.problem = problem
        self.topology = topology
        self.lib = importlib.import_module('.'.join(['systems', self.problem, self.topology]))
        # self.config = config
        self.config = {
            'network': {
                'class': 'VGG',
                'params': {
                    'config': 'VGG19',
                    'bn': True
                },
                'quantize': {
                    'recipe': 'recipeA',
                    'params': {
                        "STE": {
                            "n_levels": 255,
                            "quant_start_epoch": 28
                        },
                        "INQ": {
                            "n_levels": 3,
                            "quant_init_method": "uniform-l2-opt",
                            "quant_strategy": "magnitude"
                        }
                    }
                }
            }
        }
        import torch
        self.hw_cfg = dict()
        self.hw_cfg['device'] = torch.device('cpu')
        from manager.assistants import get_network
        self.net = get_network(self)
        ckpt = torch.load('/usr/scratch2/vilan2/spmatteo/QuantLab/problems/ImageNet/logs/exp023/fold0/saves/epoch0090.ckpt', map_location='cpu')
        self.net.load_state_dict(ckpt['network'])
        self.net.eval()

        for n in self.net.named_modules():  # TODO: started should ALWAYS be changed to 'started' when module is in 'eval' mode (otherwise tracing is fucked up!)
            if hasattr(n[1], 'started'):  # put STE nodes in "quantized mode"
                n[1].started = True

        # self.recipe = None

    @staticmethod
    def show_net(net):
        list_nodes(net, verbose=True)

    # def apply_recipe(self):
    #     self.qnet = getattr(self.recipe, 'quantize')(self.config['network']['quantize']['params'], copy.deepcopy(self.net))
    #
    # def load_recipe(self):
    #     try:
    #         importlib.reload(self.recipe)
    #     except TypeError:
    #         self.recipe = importlib.import_module('.'.join(['', 'quantize', self.config['network']['quantize']['recipe']]), package=self.lib.__name__)
