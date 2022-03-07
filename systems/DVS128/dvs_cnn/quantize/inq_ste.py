import quantlib.editing.lightweight as qlw
import quantlib.editing.lightweight.rules as qlr
import quantlib.algorithms as qa
from quantlib.algorithms.inq import *
from quantlib.algorithms.ste import STEActivation

from systems.DVS128.dvs_cnn import CausalConv1d
from torch import nn



__all__ = ['layers_ste_inq', 'layers_ste_inq_get_controllers']

def inq_conv2d_from_conv2d(m : nn.Conv2d, n_levels : int, quant_init_method : str = None, quant_strategy : str = "magnitude"):
    # return an INQ Conv2d with equivalent parameters as the input Conv2d.
    # weights are copied.
    new_conv = INQConv2d(in_channels=m.in_channels, out_channels=m.out_channels,
                     kernel_size=m.kernel_size, stride=m.stride,
                     padding=m.padding, dilation=m.dilation,
                     groups=m.groups, bias=m.bias is not None,
                     padding_mode=m.padding_mode,
                     quant_init_method=quant_init_method,
                     quant_strategy=quant_strategy)
    new_conv.weight.data.copy_(m.weight.data)
    if m.bias is not None:
        new_conv.bias.data.copy_(m.bias.data)
    return new_conv

def inq_conv1d_from_conv1d(m : nn.Conv1d, n_levels : int, quant_init_method : str = None, quant_strategy : str = "magnitude"):
    # return an INQ Conv2d with equivalent parameters as the input Conv2d.
    # weights are copied.
    new_conv = INQConv1d(in_channels=m.in_channels, out_channels=m.out_channels,
                     kernel_size=m.kernel_size, stride=m.stride,
                     padding=m.padding, dilation=m.dilation,
                     groups=m.groups, bias=m.bias is not None,
                     padding_mode=m.padding_mode,
                     quant_init_method=quant_init_method,
                     quant_strategy=quant_strategy)
    new_conv.weight.data.copy_(m.weight.data)
    if m.bias is not None:
        new_conv.bias.data.copy_(m.bias.data)
    return new_conv

def inq_causal_conv1d_from_causal_conv1d(m : CausalConv1d, n_levels : int, quant_init_method : str = None, quant_strategy : str = "magnitude"):
    # return an INQ Conv2d with equivalent parameters as the input Conv2d.
    # weights are copied.
    new_conv = INQCausalConv1d(in_channels=m.in_channels, out_channels=m.out_channels,
                     kernel_size=m.kernel_size, stride=m.stride,
                     dilation=m.dilation,
                     groups=m.groups, bias=m.bias is not None,
                     padding_mode=m.padding_mode,
                     quant_init_method=quant_init_method,
                     quant_strategy=quant_strategy)
    new_conv.weight.data.copy_(m.weight.data)
    if m.bias is not None:
        new_conv.bias.data.copy_(m.bias.data)
    return new_conv

_INQ_REPLACEMENTS = {nn.Conv2d : inq_conv2d_from_conv2d,
                     nn.Conv1d : inq_conv1d_from_conv1d,
                     CausalConv1d : inq_causal_conv1d_from_causal_conv1d}


def layers_ste_inq(net, config):

    filter_convs = qlr.TypeFilter(nn.Conv2d) | qlr.TypeFilter(nn.Conv1d) | qlr.TypeFilter(CausalConv1d)
    # we support only HtanH activations but check also for ReLU
    filter_acts = qlr.TypeFilter(nn.ReLU) | qlr.TypeFilter(nn.Hardtanh)
    ste_config = config["STE"]
    inq_config = config["INQ"]
#     def get_layers_conv_nodes(net):
#         net_nodes = qg.list_nodes(net, verbose=False)
#         rule = [qg.rule_linear_nodes]
#         linear_nodes = qg.find_nodes(net_nodes, rule, mix='and')
#         return linear_nodes

#     def get_layers_by_class(net, class_name):
#         net_nodes = qg.list_nodes(net, verbose=False)
#         cond = lambda n: n.__class__.__name__ == class_name
#         filt_nodes = []
#         for n in net_nodes:
#             if cond(n.module):
#                 filt_nodes.append(n)

    def add_ste_after_htanh(net, num_levels, quant_start_epoch):
        net_nodes = qlw.LightweightGraph.build_nodes_list(net)
        act_nodes = filter_acts(net_nodes)
        for n in act_nodes:
            # we want to make sure we have ONLY Htanh nodes in our net
            assert n.module.__class__.__name__ == "Hardtanh", "Non-htanh activation found: node {} has class {}".format(n.name, type(n.module))
            ste_node = STEActivation(num_levels=num_levels, quant_start_epoch=quant_start_epoch)
            combined_node = nn.Sequential(n.module, ste_node)
            qlr.LightweightRule.replace_module(net, n.path, combined_node)

    def replace_conv_nodes_inq(net, num_levels, quant_init_method, quant_strategy):
        net_nodes = qlw.LightweightGraph.build_nodes_list(net)
        conv_nodes = filter_convs(net_nodes)
        for n in conv_nodes:
            assert isinstance(n.module, (nn.Conv2d, CausalConv1d, nn.Conv1d)), f"Bad convolution found: {type(n.module)}"
            conv_type = type(n.module)
            new_conv = _INQ_REPLACEMENTS[conv_type](n.module, num_levels, quant_init_method, quant_strategy)
            qlr.LightweightRule.replace_module(net, n.path, new_conv)

    add_ste_after_htanh(net, ste_config['n_levels'], ste_config['quant_start_epoch'])
    replace_conv_nodes_inq(net, inq_config['n_levels'], inq_config['quant_init_method'], inq_config['quant_strategy'])
    if 'verbose' in config.keys() and config['verbose']:
        print("Edited network:")
        for n in qlw.LightweightGraph.build_nodes_list(net):
            print(n)

    return net


def layers_ste_inq_get_controllers(net, config):

    net_nodes = qlw.LightweightGraph.build_nodes_list(net)

    # get STE controller
    ste_ctrl_config = config['STE']
    ste_modules = qa.ste.STEController.get_ste_modules(net_nodes)
    ste_controller = qa.ste.STEController(ste_modules, ste_ctrl_config['clear_optim_state_on_step'])

    # get INQ controller
    inq_ctrl_config = config['INQ']
    inq_modules = qa.inq.INQController.get_inq_modules(net_nodes)
    inq_controller = qa.inq.INQController(inq_modules, inq_ctrl_config['schedule'], inq_ctrl_config['clear_optim_state_on_step'])

    return [ste_controller, inq_controller]
