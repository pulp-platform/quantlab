import os
import torch.nn as nn

import quantlab.graphs.analyse as qa
from .layers import convert_h2d, convert_d2d, convert_d2h, convert_h2h
from .twn_accelerator import *


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


def compile_vgg(net, output_dir=os.path.curdir, input_type='float'):

    layers = parse_vgg(net)
    net_name = 'VGG{}'.format(len(layers))
    output_dir = os.path.join(output_dir, net_name)
    c_out_dir = os.path.join(output_dir, "C_OUT")
    # put the parameters in yet another subfolder so it can just be copied to SD card
    output_dir = os.path.join(output_dir, net_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(c_out_dir, exist_ok=True)
    tq_net = []
    fq_net = []

    #  Set up a metadata object for the whole network to collect all the information for each layer:
    #  - layer parameters (kernel size, buffer sizes, etc)
    #  - location of outputs
    #  - location of parameter binaries (weights & BN params)
    c_net = TWNAccelSequentialNet(name=net_name, out_dir=c_out_dir, init_dim=(112, 112))
    params = TWNAccelParams(blk_size=48)

    for i, (type_, layer) in enumerate(layers):
        if i < 19:
            name = net_name + '_layer{:0>2}_'.format(i+1) + type_
            export_dir = os.path.join(output_dir, name)
            os.makedirs(export_dir, exist_ok=True)
            convert_fn = globals()['convert_' + type_]
            kwargs = {'input_type': input_type} if i == 0 else {'params':params} if type_ in ['d2d', 'd2h'] else {}
            tq_net.append(convert_fn(layer, export_dir=export_dir, **kwargs))

            fq_net.append(nn.Sequential(*[n[1] for n in layer[int(i != 0):]]))
            if type_ in ['d2d', 'd2h']:
                c_layer = TWNLayer(layer, name=name, params=params)
                c_net.add_layer(c_layer)

    # Export the embedded C code to set up and run the network!
    # That's it!
    c_net.render()
    return tq_net, fq_net
