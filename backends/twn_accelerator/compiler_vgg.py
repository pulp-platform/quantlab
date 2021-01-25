import os
import torch.nn as nn
import numpy as np

import quantlab.graphs.analyse as qa
from .layers import convert_h2d, convert_d2d, convert_d2h, convert_h2h
from .twn_accelerator import *
from .acl import ACLNet, ACLTensor, ACLDequantLayer
from backends.abstract_net import QuantProperties
from copy import deepcopy
from .quantops import DequantLayer



def parse_vgg(net):

    net_nodes = qa.list_nodes(net)

    h2d_layers = []
    d2d_layers = []  # these are the only layers executed on the accelerator
    d2h_layers = []
    h2h_layers = []

    # since VGG is fully feedforward, the nodes of each layer are enclosed in-between activation functions;
    # therefore, knowing the positions of the activation layers is sufficient to parse nodes into layers
    ste_nodes = [(i, n[0]) for i, n in enumerate(net_nodes) if n[1].__class__.__name__ == 'STEActivation']
    linear_nodes = [(i, n[0]) for i, n in enumerate(net_nodes) if n[1].__class__.__name__ == 'Linear' and i > ste_nodes[-1][0]]  # assumption: 'h2h' layers are at the end of the network!

    # host-to-device layer
    h2d_layers.append(('h2d', net_nodes[0:ste_nodes[0][0] + 1]))

    # device-only layers
    for i in range(0, len(ste_nodes) - 1):
        d2d_layers.append(('d2d', net_nodes[ste_nodes[i][0]:ste_nodes[i+1][0] + 1]))

    # device-to-host layer
    d2h_layers.append(('d2h', net_nodes[ste_nodes[-1][0]:linear_nodes[0][0]]))

    # host-only layers
    for i in range(0, len(linear_nodes) - 1):
        h2h_layers.append(('h2h', net_nodes[linear_nodes[i][0] - 1:linear_nodes[i+1][0]]))
    h2h_layers.append(('h2h', net_nodes[linear_nodes[-1][0] - 1:]))  # last layer

    layers = h2d_layers + d2d_layers + d2h_layers + h2h_layers

    return layers

def zero_pad_4d_tensor(t : torch.Tensor, new_dims : tuple):
    z = torch.zeros(new_dims, dtype=t.dtype)
    s = t.shape
    z[0:s[0], 0:s[1], 0:s[2], 0:s[3]] = t
    return z

def pad_conv_layer(module : nn.Conv2d, new_in_ch : int, new_out_ch : int):
    module = deepcopy(module)
    w = module.weight.data
    old_out_ch = w.shape[0]
    s = w.shape[2:]
    s = (new_out_ch, new_in_ch) + s
    new_w = zero_pad_4d_tensor(w, s)
    module.weight.data = new_w
    if module.bias is not None:
        new_b = nn.functional.pad(module.bias.data, (0, new_out_ch-old_out_ch))
        module.bias.data = new_b
    if isinstance(module, INQConv2d):
        module.weight_frozen.data = zero_pad_4d_tensor(module.weight_frozen.data, s)

    module.in_channels = new_in_ch
    module.out_channels = new_out_ch
    return module

def compare_padded_conv_layers(orig_l : nn.Conv2d, padded_l : nn.Conv2d):
    unpadded_in_ch = orig_l.in_channels
    padded_in_ch = padded_l.in_channels
    in_pad = padded_in_ch - unpadded_in_ch
    unpadded_out_ch = orig_l.out_channels
    padded_out_ch = padded_l.out_channels
    out_pad = padded_out_ch - unpadded_out_ch

    unpadded_input = torch.rand((1, unpadded_in_ch, 7, 7))
    in_padding = torch.rand((1, in_pad, 7, 7))
    padded_input = torch.cat([unpadded_input, in_padding], dim=1)

    unpadded_output = orig_l(unpadded_input)
    padded_output = padded_l(padded_input)

    match_prepad = (torch.sum(unpadded_output != padded_output[:, :unpadded_out_ch, :, :]) == 0)
    zero_pad = (torch.sum(padded_output[:, unpadded_out_ch:, :, :] != 0) == 0)

    if not (match_prepad and zero_pad):
        print("ehh padding mismatch...")




def pad_bn_layer(module, new_out_ch):
    module = deepcopy(module)
    diff_channels = new_out_ch - module.num_features
    module.num_features = new_out_ch

    if module.affine:
        module.weight.data = nn.functional.pad(module.weight.data, (0, diff_channels))
        module.bias.data = nn.functional.pad(module.bias.data, (0, diff_channels))

    module.running_mean = nn.functional.pad(module.running_mean, (0, diff_channels))
    module.running_var = nn.functional.pad(module.running_var, (0, diff_channels))
    return module

# gets the next larger or equal integer multiple of blk_size to n_ch
def blocked_channels(n_ch, blk_size):
    return int(np.ceil(n_ch/blk_size) * blk_size)

def compile_vgg(net, output_dir=os.path.curdir, input_type='float'):

    layers = parse_vgg(net)
    net_name = 'VGG{}'.format(len(layers))
    output_dir = os.path.join(output_dir, net_name)
    c_out_dir = os.path.join(output_dir, "C_OUT")
    cpp_out_dir = os.path.join(output_dir, "CPP_OUT")
    # put the parameters in yet another subfolder so it can just be copied to SD card
    output_dir = os.path.join(output_dir, net_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(c_out_dir, exist_ok=True)
    # "true-quantized net" as it *should* be executed on the hardware
    #   - should be equivalent to FQ (but due to quantized batchNorm it's not)
    #   - is also not equivalent to what happens on HW, because the last STE layer is missing
    tq_net = []
    # "fake-quantized net" - what was trained, with full-precision batchNorm layers
    fq_net = []
    # true-to-hardware net, including the STE layer after the last accelerator layer
    export_net = []

    #  Set up a metadata object for the whole network to collect all the information for each layer:
    #  - make input tensor
    #  - add layers correctly to acl net
    c_net = TWNAccelSequentialNet(name=net_name, out_dir=c_out_dir, init_dim=(224, 224))
    params = TWNAccelParams(blk_size=48)
    cpp_net = ACLNet(name=net_name, cpp_out_folder=cpp_out_dir, param_out_folder=net_name, params=params)
    cpp_in_tensor = ACLTensor(None, 'src', (1,224,224,3), False, QuantProperties("float32"))
    for i, (type_, layer) in enumerate(layers):
        if i < 19:
            name = net_name + '_layer{:0>2}_'.format(i+1) + type_
            export_dir = os.path.join(output_dir, name)
            os.makedirs(export_dir, exist_ok=True)
            convert_fn = globals()['convert_' + type_]
            kwargs = {'input_type': input_type} if i == 0 else {'params':params} if type_ in ['d2d', 'd2h'] else {}
            if type_ == 'h2d':
                conv_idx, conv_node = [(i, m) for i, m in enumerate(layer) if isinstance(m.module, nn.Conv2d)][0]
                bn_idx, bn_node = [(i, m) for i, m in enumerate(layer) if isinstance(m.module, nn.BatchNorm2d)][0]
                bn_module = bn_node.module
                conv_module = conv_node.module
                new_conv = pad_conv_layer(conv_module, 3, 96)
                layer[conv_idx] = Node(conv_node.name, new_conv)

                def expand_1d(n, t):
                    z = torch.zeros((n), dtype=t.dtype)
                    return torch.cat((t, z))

                bn_new = pad_bn_layer(bn_module, 96)
                layer[bn_idx] = Node(bn_node.name, bn_new)

            # zero pad weights to TWN accel compatible size
            if type_ in ['d2h', 'd2d']:
                conv_idx, conv_node = [(i, m) for i, m in enumerate(layer) if isinstance(m.module, nn.Conv2d)][0]
                conv_module = conv_node.module
                new_in_ch = blocked_channels(conv_module.in_channels, 48)
                new_out_ch = blocked_channels(conv_module.out_channels, 48)
                new_conv = pad_conv_layer(conv_module, new_in_ch, new_out_ch)
                compare_padded_conv_layers(conv_module, new_conv)

                layer[conv_idx] = Node(conv_node.name, new_conv)

                bn_idx, bn_node = [(i, m) for i, m in enumerate(layer) if isinstance(m.module, nn.BatchNorm2d)][0]
                bn_module = bn_node.module
                new_bn = pad_bn_layer(bn_module, new_out_ch)
                layer[bn_idx] = Node(bn_node.name, new_bn)

            if type_ == 'h2h':
                # in the first h2h layer, we have "too many" channels due to the channel blocking - sidestep this by adding zero weights for the unneeded ones
                if len(interm_tensor.shape) != 1:
                    linear_weights = layer[1].module.weight.data.clone().detach()
                    linear_weights = linear_weights.reshape(4096, 512, 7,7)
                    zero_weights = torch.zeros(4096, 16, 7, 7, dtype=linear_weights.dtype)
                    linear_weights = torch.cat([linear_weights, zero_weights], dim=1)
                    linear_weights = linear_weights.reshape([4096, -1])
                    linear_layer = layer[1].module
                    new_linear_layer = deepcopy(linear_layer)
                    new_linear_layer.in_features = 528*7*7
                    new_linear_layer.weight.data = linear_weights
                    layer[1] = Node(layer[1].name, new_linear_layer)

            converted_layer = convert_fn(layer, export_dir=export_dir, **kwargs)
            export_layer = deepcopy(converted_layer)
            if type_ == 'h2d':
                # no BN layer; it's folded into the convolution
                # add_layer needs a list of nodes
                # discard the STEInteger layer; we need a "unity STE" layer here
                converted_nodes = [Node("layer_{}".format(i+1), converted_layer[i]) for i in range(len(converted_layer)-1)]
                interm_tensor = cpp_net.add_layer(converted_nodes, cpp_in_tensor, name, False)
                # need to add the STE layer manually
                # abs_max_value = 127 produces a "Cast to int8" layer
                ste = STEActivation(255)
                ste.abs_max_value.data = torch.tensor(127.0)
                # TODO parse QuantLayers correctly
                interm_tensor = cpp_net.add_layer(ste, interm_tensor, name+"_cast", False)

            if type_ in ['d2d', 'd2h']:
                c_layer = TWNLayer(layer, name=name, params=params)
                c_net.add_layer(c_layer)
                interm_tensor = cpp_net.add_layer(layer, interm_tensor, name, False)

            if type_ == 'd2h':
                # for the last layer, we need to add a dequantLayer manually.
                # TODO train a new vgg19 with the last STE in place!
                last_ste = STEActivation(num_levels=255)
                last_ste.abs_max_value.data = layer[0].module.abs_max_value.data.clone().detach()
                dequant = ACLDequantLayer(last_ste, interm_tensor, name+"_dequant", False)
                interm_tensor = cpp_net.add_layer(dequant, interm_tensor, name, False)
                # need to flatten
                flatten_node = Node('flatten', nn.Flatten())
                layer.append(flatten_node)
                converted_layer.append(nn.Flatten())
                # the "export_layer" list gets the last STE layer
                export_layer.append(nn.Flatten())
                dequant_module = DequantLayer(last_ste.abs_max_value.data.item(), 8)
                export_layer.append(dequant_module)
                # the "converted_layer" also needs this because the dequantization is no longer folded into the BN
                converted_layer.append(dequant_module)
            if type_ == 'h2h':
                # discard adaptiveAvgPool and dropout layers
                layers_in = [l for l in layer if not layer_has_modules(l, [nn.Dropout, nn.AdaptiveAvgPool2d])]
                interm_tensor = cpp_net.add_layer(layers_in, interm_tensor, name, i==18)
            export_net.append(nn.Sequential(*export_layer))
            fq_net.append(nn.Sequential(*[n[1] for n in layer[int(i != 0):]]))
            tq_net.append(nn.Sequential(*converted_layer))

    # Export the embedded C/C++ code to set up and run the network!
    c_net.render()
    cpp_net.render()

    return tq_net, fq_net, export_net, output_dir
