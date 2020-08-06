import os
import torch.nn as nn

import quantlab.graphs.analyse as qa
from .layers import convert_h2d, convert_d2d, convert_d2h, convert_h2h


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


def compile_vgg(net, output_dir=os.path.curdir):

    layers = parse_vgg(net)
    output_dir = os.path.join(output_dir, 'VGG{}'.format(len(layers)))
    os.makedirs(output_dir, exist_ok=True)

    tq_net = []
    fq_net = []

    for i, (type_, layer) in enumerate(layers):
        if i < 19:

            export_dir = os.path.join(output_dir, 'layer{:0>2}_'.format(i+1) + type_)
            os.makedirs(export_dir, exist_ok=True)

            convert_fn = globals()['convert_' + type_]
            tq_net.append(convert_fn(layer, export_dir=export_dir))

            fq_net.append(nn.Sequential(*[n[1] for n in layer[int(i != 0):]]))

    return tq_net, fq_net
