import os
from copy import deepcopy
from pathlib import Path

from typing import Union

import torch
from torch import nn

from mako.template import Template

#these imports are an atrocious hackfest, I'm sorry
import backends
from backends.abstract_net import AbstractNet, AbstractOperator, AbstractTensor, QuantProperties

from .twn_accelerator import TWNAccelParams, TWNLayer
from .layers import node_is_module

from quantlab.algorithms.ste import STEActivation
from quantlab.algorithms.inq import INQConv2d
from quantlab.graphs import Node


# ASSUMPTIONS:
# - constructors are called with NHWC shapes
# - layer modules have NCHW parameters (e.g. weights)
def expand_tuple(module, attr):
    # expand a layer parameter (e.g. padding, dilation, stride...) to a tuple
    # if it isn't one already.
    # e.g. pass a Conv2d layer and attr="kernel_size" to get (3, 3) kernel
    # size even if module.kernel_size == 3
    n_dims = int(module.__class__.__name__[-2])
    p = module.__getattribute__(attr)
    if type(p) is not tuple:
        p = (p,) * n_dims
    return p

def nchw_to_nhwc(t : torch.Tensor):
    if t.ndim == 4:
        return t.permute(0, 2, 3, 1)
    if t.ndim == 3:
        return t(1, 2, 0)
    raise NotImplementedError("Can't permute tensor with {} dimensions...".format(t.ndim))

def nhwc_to_nchw(t : torch.Tensor):
    if t.ndim == 4:
        return t.permute(0, 3, 1, 2)
    if t.ndim == 3:
        return t.permute(2, 0, 1)
    raise NotImplementedError("Can't permute tensor with {} dimensions...".format(t.ndim))


class ACLTensor(AbstractTensor):
    def __init__(self, data : torch.Tensor, name : str, shape : tuple, is_param : bool, qp : QuantProperties,
                 out_folder : str = None, out_file : str = None, alignment : int = 0):
        super(ACLTensor, self).__init__(data, name, shape, is_param, qp)
        if out_folder:
            if out_file:
                self.out_file = os.path.join(out_folder, out_file)
            else:
                self.out_file = os.path.join(out_folder, name)
        self.alignment = alignment

    @property
    def tot_size(self):
        # total number of elements
        size = 1
        for el in self.shape:
            size *= el
        return size

    @property
    def c_type(self):
        if self.qp.dtype == "float32":
            return "float"
        # int8, int4, int2 will be implemented as (packed) int8s in C
        if self.qp.dtype in  ["int8", "int4", "int2"]:
            return "int8_t"
        raise NotImplementedError("Unsupported datatype: {}".format(self.qp.dtype))

    @property
    def acl_datatype(self):
        if self.qp.dtype == "float32":
            return "DataType::F32"
        if self.qp.dtype == "int8":
            return "DataType::QASYMM8_SIGNED"
        raise NotImplementedError("Unsupported datatype: {}".format(self.qp.dtype))

    @property
    def is_fixedp(self):
        return not (self.qp.dtype == "float32")

    @property
    def shape_str(self):
        s = "{}".format(self.shape[0])
        for el in self.shape[1:]:
            s = "{}, ".format(el) + s
        return s



class ACLConv2d(AbstractOperator):
    def __init__(self, nodes : Union[nn.Conv2d, list], in_tensor : ACLTensor, name : str, is_last : bool = False, out_folder : str = "."):
        super(ACLConv2d, self).__init__()
        assert in_tensor.shape is not None and None not in in_tensor.shape, "Invalid shape for in_tensor!"
        self.name = name
        self.add_input(in_tensor)
        self.activate = False
        if isinstance(nodes, list):
            assert len(nodes) in [1, 2], "can only take lists of 1 or 2 modules!"
            assert isinstance(nodes[0], nn.Conv2d), "first node in list must be Conv2d"
            if len(nodes) == 2:
                assert isinstance(nodes[1], nn.ReLU), "second node must be ReLU!"
                self.activate = True
            module = nodes[0]
        elif isinstance(nodes, nn.Conv2d):
            module = nodes


        self.module = deepcopy(module).cpu()
        self.out_folder = os.path.join(out_folder, name)
        Path(self.out_folder).mkdir(parents=True, exist_ok=True)
        wt_torch = module.weight.data
        # need to transpose to NHWC
        wt_torch = nchw_to_nhwc(wt_torch)
        wt_tensor = ACLTensor(wt_torch, name+'_weight', None, True, QuantProperties("float32"), self.out_folder, "weight")
        self.add_param(wt_tensor)
        if module.bias is not None:
            bias_torch = module.bias
            bias_tensor = ACLTensor(bias_torch, name+'_bias', None, True,  QuantProperties("float32"), self.out_folder, "bias")
            self.add_param(bias_tensor)

        self.cpp_namespace = "arm_compute"
        self.acl_type = "NEConvolutionLayer"
        self.constructor_args = "mm_layers"

        dummy_input = torch.zeros(in_tensor.shape, dtype=torch.float64)
        dummy_input = nhwc_to_nchw(dummy_input)
        cpu_module = deepcopy(module).cpu()
        with torch.no_grad():
            dummy_output = cpu_module(dummy_input)
        dummy_output = nchw_to_nhwc(dummy_output)
        out_shape = tuple(dummy_output.shape)
        out_name = "dst" if is_last else name+"_out"
        out_tensor = ACLTensor(None, out_name, out_shape, False, QuantProperties("float32"))
        self.add_output(out_tensor)
        self.padding = expand_tuple(module, "padding")
        self.stride = expand_tuple(module, "stride")

    @property
    def qualified_type(self):
        return self.cpp_namespace+"::"+self.acl_type

    @property
    def pad_stride_info(self):
        return self.cpp_namespace+"::"+"PadStrideInfo({}/*x stride*/, {}/*y stride*/, {}/*x padding*/, {}/*y padding*/)".format(self.stride[1], self.stride[0], self.padding[1], self.padding[0])

    @property
    def configure_args(self):
        # need to pass:
        # 1. input tensor
        # 2. weight tensor
        # 3. biases (if present)
        # 4. output tensor
        # 5. PadStrideInfo
        bias_name = "NULL" if len(self.parameters) == 1 else self.parameters[1].name
        config_args = "&{}, &{}, &{}, &{}, {}".format(self.inputs[0].name, self.parameters[0].name, bias_name, self.outputs[0].name, self.pad_stride_info)
        if self.activate:
            # if we want to fuse a ReLU with the convolution, we have to also
            dil = expand_tuple(self.module, 'dilation')
            config_args += ', WeightsInfo(), Size2D({}/*x dilation*/, {}/*y dilation*/), ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)'.format(dil[1], dil[0])
        return config_args


class ACLLinear(AbstractOperator):
    def __init__(self, nodes : Union[nn.Linear, list], in_tensor : ACLTensor, name : str, is_last : bool = False, out_folder : str = "."):
        super(ACLLinear, self).__init__()


        self.out_folder = os.path.join(out_folder, name)
        Path(self.out_folder).mkdir(parents=True, exist_ok=True)

        self.name = name
        self.activate = False
        if isinstance(nodes, nn.Linear):
            module = nodes
        elif isinstance(nodes, list):
            assert len(nodes) in [1,2], "nodes list must have length 1 (linear layer only) or 2 (linear + relu)"
            module = nodes[0]
            if len(nodes) == 2:
                assert isinstance(nodes[1], nn.ReLU), "Second node must be ReLU"
                self.activate = True

        self.module = module
        assert in_tensor.numel == module.in_features, "Input tensor to Linear Layer should have {} elements, but has {}".format(
                module.in_features, in_tensor.numel)
        self.add_input(in_tensor)
        self.is_last = is_last

        wt_shape = (module.out_features, module.in_features)
        wt_tensor = ACLTensor(module.weight.data.detach().clone(), name+"_weight", wt_shape, True, QuantProperties("float32"), self.out_folder, "weight")
        self.add_param(wt_tensor)

        self.cpp_namespace = "arm_compute"
        self.acl_type = "NEFullyConnectedLayer"
        self.constructor_args = "mm_layers"

        if module.bias is not None:
            bias_tensor = ACLTensor(module.bias.data.detach().clone(), name+"_bias", (module.out_features,), True, QuantProperties("float32"), self.out_folder, "bias")
            self.add_param(bias_tensor)

        out_name = "dst" if is_last else name+"_out"
        out_tensor = ACLTensor(None, out_name, (module.out_features,), False, QuantProperties("float32"))
        self.add_output(out_tensor)

    @property
    def qualified_type(self):
        return self.cpp_namespace+"::"+self.acl_type

    @property
    def configure_args(self):
        # need to pass:
        # 1. input tensor
        # 2. weight tensor
        # 3. bias tensor (if present)
        # 4. output tensor
        bias_name = "NULL" if len(self.parameters) == 1 else self.parameters[1].name
        config_args = "&{}, &{}, &{}, &{}".format(self.inputs[0].name, self.parameters[0].name, bias_name, self.outputs[0].name)
        # really ugly.
        if self.activate:
            config_args += ", FullyConnectedLayerInfo{DataLayout::NCHW, true, false, false, false, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU)}"
        return config_args


class ACLMaxPool2d(AbstractOperator):
    def __init__(self, module : nn.MaxPool2d, in_tensor : ACLTensor, name : str, is_last : bool = False):
        super(ACLMaxPool2d, self).__init__()
        assert in_tensor.shape is not None and None not in in_tensor.shape, "Unsupported input tensor shape: {}".format(in_tensor.shape)

        self.kernel_size = expand_tuple(module, "kernel_size")

        self.name = name
        self.module = module
        self.is_last = is_last
        self.add_input(in_tensor)

        self.cpp_namespace = "arm_compute"
        self.acl_type = "NEPoolingLayer"
        self.constructor_args = "mm_layers"

        dummy_input = torch.zeros(in_tensor.shape)
        dummy_input = nhwc_to_nchw(dummy_input)
        cpu_module = deepcopy(module).cpu()
        with torch.no_grad():
            dummy_output = cpu_module(dummy_input)
        dummy_output = nchw_to_nhwc(dummy_output)
        out_shape = tuple(dummy_output.shape)
        out_name = "dst" if is_last else name+"_out"
        out_tensor = ACLTensor(None, out_name, out_shape, False, QuantProperties("float32"))
        self.add_output(out_tensor)
        self.padding = expand_tuple(module, "padding")
        self.stride = expand_tuple(module, "stride")

    @property
    def qualified_type(self):
        return self.cpp_namespace+"::"+self.acl_type

    @property
    def pad_stride_info(self):
        return "PadStrideInfo({}/*x stride*/, {}/*y stride*/, {}/*x padding*/, {}/*y padding*/)".format(self.stride[1], self.stride[0], self.padding[1], self.padding[0])

    @property
    def pooling_info(self):
        pooling_type = "PoolingType::MAX"
        pool_size = "Size2D({}/*width*/, {}/*height*/)".format(self.kernel_size[1], self.kernel_size[0])
        layout = "DataLayout::NHWC"
        return "PoolingLayerInfo({}, {}, {}, {})".format(pooling_type, pool_size, layout, self.pad_stride_info)

    @property
    def configure_args(self):
        # need to pass:
        # 1. input tensor
        # 2. output tensor
        # 3. pooling info
        return "&{}, &{}, &{}".format(self.inputs[0].name, self.outputs[0].name, self.pooling_info)

class ACLQuantLayer(AbstractOperator):
    def __init__(self, module : STEActivation, in_tensor : ACLTensor, name : str, is_last : bool = False):
        # only 255-step symmetric int8 quantization supported.
        super(ACLQuantLayer, self).__init__()
        assert in_tensor.dtype == "float32", "Input to QuantLayer must be FP32!"
        assert in_tensor.shape is not None and None not in in_tensor.shape

        self.is_last = is_last
        self.name = name

        self.add_input(in_tensor)

        out_step_size = module.abs_max_value.data.clone().detach().item()/127
        out_qp = QuantProperties("int8", out_step_size, 255)
        out_name = "dst" if is_last else name+"_out"
        # alignment of out tensor needs to be 16 as we (presumably) want to use it with the TWN accelerator
        out_tensor = ACLTensor(None, out_name, in_tensor.shape, False, out_qp, alignment=16)
        self.add_output(out_tensor)

        self.acl_type = "NEQuantizationLayer"
        self.cpp_namespace = "arm_compute"
        self.constructor_args = ""

    @property
    def qualified_type(self):
        return self.cpp_namespace+"::"+self.acl_type

    @property
    def configure_args(self):
        return "&{}, &{}".format(self.inputs[0].name, self.outputs[0].name)


class ACLDequantLayer(AbstractOperator):
    def __init__(self, module : STEActivation, in_tensor : ACLTensor, name : str, is_last : bool = False):
        # only 255-step symmetric int8 quantization supported.
        super(ACLDequantLayer, self).__init__()
        assert in_tensor.dtype == "int8", "Input to DequantLayer must be int8!"
        assert in_tensor.shape is not None and None not in in_tensor.shape

        self.is_last = is_last
        self.name = name

        self.add_input(in_tensor)

        out_step_size = module.abs_max_value.data.clone().detach().item()/127
        out_qp = QuantProperties("float32")
        out_name = "dst" if is_last else name+"_out"
        out_tensor = ACLTensor(None, out_name, in_tensor.shape, False, out_qp, None)
        self.add_output(out_tensor)

        self.acl_type = "NEDequantizationLayer"
        self.cpp_namespace = "arm_compute"
        self.constructor_args = ""

    @property
    def qualified_type(self):
        return self.cpp_namespace+"::"+self.acl_type

    @property
    def configure_args(self):
        return "&{}, &{}".format(self.inputs[0].name, self.outputs[0].name)

class ACLCastLayer(AbstractOperator):
    def __init__(self, in_tensor : ACLTensor, name : str, is_last : bool = False):
        super(ACLCastLayer, self).__init__()
        assert in_tensor.shape is not None and None not in in_tensor.shape

        self.is_last = is_last
        self.name = name

        self.add_input(in_tensor)
        if in_tensor.dtype == "float32":
            out_dtype = "int8"
        elif in_tensor.dtype == "int8":
            out_dtype = "float32"
        out_qp = QuantProperties(out_dtype)
        out_name = name+"_out"
        out_tensor = ACLTensor(None, out_name, in_tensor.shape, False, out_qp, None)
        self.add_output(out_tensor)

        self.acl_type = "NECast"
        self.cpp_namespace = "arm_compute"
        self.constructor_args = ""

    @property
    def qualified_type(self):
        return self.cpp_namespace+"::"+self.acl_type

    @property
    def configure_args(self):
        return "&{}, &{}, {}".format(self.inputs[0].name, self.outputs[0].name, "ConvertPolicy::SATURATE")


class ACLTWNLayer(AbstractOperator):
    def __init__(self, nodes : list, in_tensor : ACLTensor, name : str, params : TWNAccelParams, is_last : bool = False, out_folder : str = "."):
        # TWN layer takes a bunch of nodes, not just a single module.
        # this is because one ACLTWNLayer represents a "stack" of operations.
        super(ACLTWNLayer, self).__init__()
        self.is_last = is_last
        assert in_tensor.dtype == "int8", "Input to TWNLayer must be int8!"
        assert in_tensor.shape is not None and None not in in_tensor.shape
        self.add_input(in_tensor)

        # TODO this is a nasty hack, maybe make it nicer
        self.layer_info = TWNLayer(nodes, name+"_info", params)

        # haha...
        try:
            step_size = self.layer_info.get_out_ste().abs_max_value.data.clone().detach().item() / 127
        except:
            # if there is no out_ste, we take the step size from in_ste
            step_size = self.layer_info.get_in_ste().abs_max_value.data.clone().detach().item() / 127
        out_qp = QuantProperties("int8", step_size, 255)
        out_name = "dst" if is_last else name+"_out"
        out_shape = self.layer_info.get_out_shape(in_tensor.shape)
        out_tensor = ACLTensor(None, out_name, out_shape, False, out_qp, None, alignment=16)

        self.add_output(out_tensor)
        # add gamma and beta (to template)
        self.name = name
        self.out_folder = os.path.join(out_folder, name)
        Path(self.out_folder).mkdir(parents=True, exist_ok=True)

        self.acl_type = "TWNAccelLayer"
        self.cpp_namespace = "arm_compute"

        self.constructor_args = "twn_cfg, dma_cfg"

    @property
    def qualified_type(self):
        return self.cpp_namespace+"::"+self.acl_type

    @property
    def configure_args(self):
        # residual always NULL
        return "&{}, &{}, NULL, &{}".format(self.inputs[0].name, self.outputs[0].name, self.layer_info.layer_name)

    @property
    def weights_filename(self):
        return os.path.join(self.out_folder, "weight")

    @property
    def gamma_filename(self):
        return os.path.join(self.out_folder, "gamma")

    @property
    def beta_filename(self):
        return os.path.join(self.out_folder, "beta")

    @property
    def line_width(self):
        return self.inputs[0].shape[-2]

    @property
    def n_lines(self):
        return self.inputs[0].shape[-3]


class ACLNet(AbstractNet):
    def __init__(self, name : str, params : TWNAccelParams, cpp_out_folder : str, param_out_folder : str):
        super(ACLNet, self).__init__(name)
        self.template_dir = os.path.join(os.path.dirname(backends.twn_accelerator.__file__), "templates")
        self.cpp_out_folder = os.path.join(cpp_out_folder, name)
        # tensors don't handle their own exporting (yet) so param_out_folder is just from the perspective of
        # the final sd card root
        self.param_out_folder = param_out_folder
        self.params = params
        self.header_template = os.path.join(self.template_dir, "acl_net.h.mako")
        self.net_template = os.path.join(self.template_dir, "acl_net.cpp.mako")
        self.twn_header_template = os.path.join(self.template_dir, "twn_layer_defs.h.mako")
        self.header_fn = os.path.join(self.cpp_out_folder, self.name+"_net.h")
        self.code_fn = os.path.join(self.cpp_out_folder, self.name+"_net.cpp")
        self.twn_header_fn = os.path.join(self.cpp_out_folder, self.name+"_twn.h")

    def add_layer(self, l : Union[Node, nn.Module, AbstractOperator, list], in_tensor : ACLTensor, name : str, is_last : bool = False):
        op = None
        if isinstance(l, Node):
            l = l.module
        if isinstance(l, nn.Module):
            if isinstance(l, nn.Conv2d):
                op = ACLConv2d(l, in_tensor, name, is_last, self.param_out_folder)
            elif isinstance(l, nn.Linear):
                op = ACLLinear(l, in_tensor, name, is_last, self.param_out_folder)
            elif isinstance(l, nn.MaxPool2d):
                op = ACLMaxPool2d(l, in_tensor, name, is_last)
            elif isinstance(l, STEActivation):
                if l.abs_max_value.data.item() == 127.0:
                    op = ACLCastLayer(in_tensor, name, is_last)
                else:
                    op = ACLQuantLayer(l, in_tensor, name, is_last)
        elif isinstance(l, list):
            l, out_tensor = self.parse_layer_list(l, in_tensor, name, is_last)
            assert len(l) == 0, "Only a single pass of parse_layer_list supported for now..."
        elif isinstance(l, AbstractOperator):
            op = l
        else:
            raise TypeError("Bad input layer to ACLNet.add_layer: {}".format(type(l)))

        if op is not None:
            super(ACLNet, self).add_layer(op)
            out_tensor = op.outputs[0]
        return out_tensor

    def parse_layer_list(self, l : list, in_tensor : ACLTensor, name : str, is_last : bool = False):

        # take modules out of Nodes
        l = [m.module if isinstance(m, Node) else m for m in l]
        # if INQConv is in the list, assume we are dealing with a TWN layer
        if any(isinstance(n, INQConv2d) for n in l):
            op = ACLTWNLayer(l, in_tensor, name, self.params, is_last, self.param_out_folder)
            super(ACLNet, self).add_layer(op)
            return [], op.outputs[0]
        def fused_relu(l : list):
            return len(l) > 1 and isinstance(l[1], nn.ReLU)

        if isinstance(l[0], nn.Conv2d):
            if fused_relu(l):
                n = 2
            else:
                n = 1
            layer = l[0:n]
            op = ACLConv2d(layer, in_tensor, name, is_last and len(l)==n, self.param_out_folder)
        elif isinstance(l[0], nn.Linear):
            if fused_relu(l):
                n = 2
            else:
                n = 1
            layer = l[0:n]
            op = ACLLinear(layer, in_tensor, name, is_last and len(l)==n, self.param_out_folder)
        elif isinstance(l[0], nn.MaxPool2d):
            n = 1
            layer = l[0]
            op = ACLMaxPool2d(layer, in_tensor, name, is_last and len(l)==n)
        elif isinstance(l[0], nn.BatchNorm2d):
            assert(False), "Please fold & remove BatchNorm2d"
        else:
            raise TypeError("Unsupported layer: {}".format(type(l[0])))
        super(ACLNet, self).add_layer(op)
        return l[n:], op.outputs[0]

    @property
    def twn_layers(self):
        return [l for l in self.layers if isinstance(l, ACLTWNLayer)]

    def render(self):
        Path(self.cpp_out_folder).mkdir(parents=True, exist_ok=True)
        h_template = Template(filename=self.header_template)
        cpp_template = Template(filename=self.net_template)
        twn_template = Template(filename=self.twn_header_template)
        for f, t in [(self.header_fn, h_template), (self.code_fn, cpp_template), (self.twn_header_fn, twn_template)]:
            with open(f, 'w') as fh:
                fh.write(t.render(n=self))

    @property
    def header_guard(self):
        return "_"+os.path.basename(self.header_fn).replace(".", "_").upper()

    @property
    def twn_header_guard(self):
        return "_"+os.path.basename(self.twn_header_fn).replace(".", "_").upper()

    @property
    def intermediate_tensors(self):
        d = self.data_tensors
        return [t for t in d if not t.name in ["dst", "src"]]

    @property
    def managed_tensors(self):
        d = self.data_tensors
        return [t for t in d if t.name != "src"]
