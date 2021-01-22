import os

import torch
import torch.nn as nn

import backends

from quantlab.algorithms.inq import INQConv2d, INQConv1d
from quantlab.algorithms.ste import STEActivation
from quantlab.graphs.analyse import Node
from .layers import layer_has_modules
from mako.template import Template

class TWNAccelParams:
    def __init__(self, blk_size : int = 48):
        self.blk_size = blk_size

    # how many channel blocks are needed to store the number of channels
    def ch_blks(self, n_ch : int):
        return (n_ch-1)//self.blk_size+1

class TWNAccelSequentialNet:
    def __init__(self, name : str, out_dir : str, init_dim : tuple = None):
        self.name = name
        self.out_dir = out_dir
        self.layers = []
        #holy hell this is ugly!! how do I make it butifel??
        template_dir = os.path.join(os.path.dirname(backends.twn_accelerator.__file__), 'templates')
        self.get_layer_template = os.path.join(template_dir, 'get_layer')
        self.get_net_template = os.path.join(template_dir, 'get_net')
        # init_dim has the format (H, W)
        self.init_dim = init_dim

    def header_template(self, t):
        return t+".h.mako"

    def code_template(self, t):
        return t+".c.mako"

    @property
    def n_layers(self):
        return len(self.layers)

    @property
    def get_net_fn(self):
        return "get_net_" + self.name

    @property
    def get_net_header(self):
        return self.get_net_fn + ".h"
    @property
    def get_net_code(self):
        return self.get_net_fn + ".c"

    @property
    def header_macro(self):
        return '_'+self.get_net_header.replace('.', '_').upper()

    def add_layer(self, l):
        self.layers.append(l)

    def render(self):
        get_layer_h = Template(filename=self.header_template(self.get_layer_template))
        get_layer_c = Template(filename=self.code_template(self.get_layer_template))
        get_net_h = Template(filename=self.header_template(self.get_net_template))
        get_net_c = Template(filename=self.code_template(self.get_net_template))
        for l in self.layers:
            outfile_c = os.path.join(self.out_dir, l.get_layer_code)
            outfile_h = os.path.join(self.out_dir, l.get_layer_header)
            with open(outfile_c, 'w') as fh:
                fh.write(get_layer_c.render(l=l, n=self, accel_params=l.params))
            with open(outfile_h, 'w') as fh:
                fh.write(get_layer_h.render(l=l))

        outfile_c = os.path.join(self.out_dir, self.get_net_code)
        outfile_h = os.path.join(self.out_dir, self.get_net_header)
        with open(outfile_c, 'w') as fh:
            fh.write(get_net_c.render(n=self))
        with open(outfile_h, 'w') as fh:
            fh.write(get_net_h.render(n=self))


class TWNLayer:
    # this class is used to fill the get_layer.c and get_layer.h templates
    # for a specific TWN accelerator layer
    def __init__(self, layer_nodes : list, name : str, params : TWNAccelParams):
        self.layer_name = name
        self.layer_nodes = layer_nodes
        self.params = params

    @property
    def pool_type(self):
        return "MAXPOOL" if layer_has_modules(self.layer_nodes, nn.MaxPool2d) else "AVGPOOL" if layer_has_modules(self.layer_nodes, nn.AvgPool2d) else "NO_POOL"

    @property
    def linebuf_order(self):
        conv_layer = self.get_conv_layer()
        if not conv_layer:
            assert False, "No Conv Layer found - can't say if layer has stride two"
        assert conv_layer.stride in [(1,1), (2,2)], "Unsupported Stride: {}".format(conv_layer.stride)
        stride_two = (conv_layer.stride == 2)
        pooling = layer_has_modules(self.layer_nodes, [nn.MaxPool2d, nn.AvgPool2d])
        if pooling and stride_two:
            assert False, "Pooling and stride_two in the same layer currently not supported!"
        return "POOLING" if pooling else "STRIDE_TWO" if stride_two else "REGULAR"

    @property
    def get_layer_fn(self):
        return "get_" + self.layer_name

    @property
    def get_layer_header(self):
        return self.get_layer_fn+".h"

    @property
    def get_layer_code(self):
        return self.get_layer_fn+".c"

    @property
    def header_macro(self):
        return '_'+self.get_layer_header.replace('.', '_').upper()

    def get_conv_layer(self):
        conv_layer = None
        n_conv_layers = 0
        for n in self.layer_nodes:
            if isinstance(n.module, INQConv2d) or isinstance(n.module, INQConv1d):
                conv_layer = n.module
                n_conv_layers += 1
        if n_conv_layers > 1:
            print("Warning: found more than 1 conv2d layer in get_conv_layer!")
        return conv_layer

    @property
    def ste_nodes(self):
        ste = [n for n in self.layer_nodes if isinstance(n.module, STEActivation)]
        return ste
    # there should be 2 STE layers in self.layer_nodes: one for the
    # quantization of the input and one for the quantization of the output
    def get_in_ste(self):
        return self.ste_nodes[0].module

    def get_out_ste(self):
        return self.ste_nodes[1].module

    @property
    def n_in_blk(self):
        conv_layer = self.get_conv_layer()
        if not conv_layer:
            assert False, "No Conv Layer found - n_in_blk can't be determined!"
        return self.params.ch_blks(conv_layer.in_channels)

    @property
    def n_out_blk(self):
        conv_layer = self.get_conv_layer()
        if not conv_layer:
            assert False, "No Conv Layer found - n_out_blk can't be determined!"
        return self.params.ch_blks(conv_layer.out_channels)

    @property
    def n_in_ch(self):
        return self.params.blk_size*self.n_in_blk

    @property
    def n_out_ch(self):
        return self.params.blk_size*self.n_out_blk

    @property
    def K(self):
        conv_layer = self.get_conv_layer()
        if not conv_layer:
            assert False, "No Conv Layer found - K can't be determined!"
        k = conv_layer.kernel_size
        if len(k) == 2:
            if not k[0] == k[1]:
                assert False, "non-square kernel size - not supported!"
        else:
            if not k[0] == 1:
                assert False, "Conv1d only supported with K=1!"
        return k[0]

    @property
    def K_text(self):
        if self.K == 1:
            return "ONE"
        if self.K == 3:
            return "THREE"
        if self.K == 5:
            return "FIVE"
        if self.K == 7:
            return "SEVEN"
        assert False, "Invalid Kernel Size: {}".format(self.K)

    @property
    def weight_buf_size(self):
        return self.n_in_blk*self.n_out_blk*self.params.blk_size**2*self.K**2//4

    @property
    def weight_buf_shape(self):
        return (self.n_out_blk*self.params.blk_size, self.K, self.K, self.n_in_blk*self.params.blk_size)

    def get_out_shape(self, in_shape : tuple):
        # returns HWC output shape given HWC input shape
        assert len(in_shape) in [3, 4],  "in_shape must describe 3D tensor!"
        out_shape = [in_shape[0], 0, 0, self.n_out_ch]
        div = 1 if self.pool_type == "NO_POOL" and self.linebuf_order == "REGULAR" else 2
        for k in range(2):
            out_shape[k+1] = in_shape[k+1]//div
        if len(in_shape) == 3:
            out_shape = out_shape[1:]
        return tuple(out_shape)



    @property
    def relu(self):
        relu = layer_has_modules(self.layer_nodes, [nn.ReLU])
        return "true" if relu else "false"

    @property
    def weights_filename(self):
        return os.path.join(self.layer_name, "weight")

    @property
    def beta_filename(self):
        return os.path.join(self.layer_name, "beta")

    @property
    def gamma_filename(self):
        return os.path.join(self.layer_name,"gamma")

    @property
    def weights_varname(self):
        return self.layer_name + "_weights"

    @property
    def beta_varname(self):
        return self.layer_name + "_beta"

    @property
    def gamma_varname(self):
        return self.layer_name + "_gamma"

