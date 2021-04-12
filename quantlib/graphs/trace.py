import networkx as nx
import os

import quantlib.graphs as qg
from quantlib.graphs.utils import draw_graph


__TRACES_DIR__ = os.path.join(os.path.dirname(os.path.realpath(qg.__file__)), 'onnxtraces')


def trace_pytorch_module(mod, dummy_input):

    _, G = qg.morph.get_onnx_graph(mod, dummy_input)

    # label each node with ONNX type
    nodes_2_types = {k: v.nodetype for k, v in nx.get_node_attributes(G, 'payload').items()}
    nx.set_node_attributes(G, nodes_2_types, 'type')

    # release handles on 'torch._C.Node' and 'torch._C.Value' objects
    for n in G.nodes:
        del G.nodes[n]['payload']

    return G


def make_mod_trace_dir(algorithm, mod_name):

    mod_trace_dir = os.path.join(__TRACES_DIR__, algorithm, mod_name)
    if not os.path.isdir(mod_trace_dir):
        os.makedirs(mod_trace_dir, exist_ok=True)

    return mod_trace_dir


def store_trace(G, mod_trace_dir):
    nx.write_gpickle(G, os.path.join(mod_trace_dir, 'onnxtrace'))


def load_traces_library():

    traces_library = dict()

    for algorithm in os.listdir(__TRACES_DIR__):
        for mod_name in os.listdir(os.path.join(__TRACES_DIR__, algorithm)):
            k = mod_name
            v = nx.read_gpickle(os.path.join(__TRACES_DIR__, algorithm, mod_name, 'onnxtrace'))
            traces_library[k] = v

    return traces_library


####################################
## GENERIC PARAMETERS FOR TRACING ##
####################################

_batch_size        = 1
_n_input_channels  = 8
_n_output_channels = 8
_dim1              = 32
_dim2              = 32
_dim3              = 32
_kernel_size       = 3
_stride            = 1
_padding           = 1


#############################
## PYTORCH MODULES TRACING ##
#############################

def pytorch():

    import torch
    import torch.nn as nn

    library = 'PyTorch'

    mod_AdaptiveAvgPool1d = nn.AdaptiveAvgPool1d((int(_dim1 / 4)))
    mod_AdaptiveAvgPool1d.eval()
    G_AdaptiveAvgPool1d = trace_pytorch_module(mod_AdaptiveAvgPool1d, torch.ones(_batch_size, _n_input_channels, _dim1))
    mod_trace_dir = make_mod_trace_dir(library, mod_AdaptiveAvgPool1d.__class__.__name__)
    store_trace(G_AdaptiveAvgPool1d, mod_trace_dir)
    draw_graph(G_AdaptiveAvgPool1d, dir=mod_trace_dir)

    mod_AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((int(_dim1 / 4), int(_dim2 / 4)))
    mod_AdaptiveAvgPool2d.eval()
    G_AdaptiveAvgPool2d = trace_pytorch_module(mod_AdaptiveAvgPool2d, torch.ones(_batch_size, _n_input_channels, _dim1, _dim2))
    mod_trace_dir = make_mod_trace_dir(library, mod_AdaptiveAvgPool2d.__class__.__name__)
    store_trace(G_AdaptiveAvgPool2d, mod_trace_dir)
    draw_graph(G_AdaptiveAvgPool2d, dir=mod_trace_dir)

    mod_AdaptiveAvgPool3d = nn.AdaptiveAvgPool3d((int(_dim1 / 4), int(_dim2 / 4), int(_dim3 / 4)))
    mod_AdaptiveAvgPool3d.eval()
    G_AdaptiveAvgPool3d = trace_pytorch_module(mod_AdaptiveAvgPool3d, torch.ones(_batch_size, _n_input_channels, _dim1, _dim2))
    mod_trace_dir = make_mod_trace_dir(library, mod_AdaptiveAvgPool3d.__class__.__name__)
    store_trace(G_AdaptiveAvgPool3d, mod_trace_dir)
    draw_graph(G_AdaptiveAvgPool3d, dir=mod_trace_dir)


#################################
## QUANTLIB ALGORITHMS TRACING ##
#################################

def quantlib():

    import torch
    import quantlib.algorithms as qa

    #########
    ## STE ##
    #########
    algorithm = 'STE'

    mod_STEActivation = qa.ste.STEActivation()
    mod_STEActivation.eval()
    G_STEActivation = trace_pytorch_module(mod_STEActivation, torch.ones((_batch_size, _n_input_channels)))
    mod_trace_dir = make_mod_trace_dir(algorithm, mod_STEActivation.__class__.__name__)
    store_trace(G_STEActivation, mod_trace_dir)
    draw_graph(G_STEActivation, dir=mod_trace_dir)

    #########
    ## INQ ##
    #########
    algorithm = 'INQ'

    mod_INQConv1d = qa.inq.INQConv1d(_n_input_channels, _n_output_channels, kernel_size=_kernel_size, stride=_stride, padding=_padding, bias=False)
    mod_INQConv1d.eval()
    G_INQConv1d = trace_pytorch_module(mod_INQConv1d, torch.ones((_batch_size, _n_input_channels, _dim1)))
    mod_trace_dir = make_mod_trace_dir(algorithm, mod_INQConv1d.__class__.__name__)
    store_trace(G_INQConv1d, mod_trace_dir)
    draw_graph(G_INQConv1d, dir=mod_trace_dir)

    mod_INQConv2d = qa.inq.INQConv2d(_n_input_channels, _n_output_channels, kernel_size=_kernel_size, stride=_stride, padding=_padding, bias=False)
    mod_INQConv2d.eval()
    G_INQConv2d = trace_pytorch_module(mod_INQConv2d, torch.ones((_batch_size, _n_input_channels, _dim1, _dim2)))
    mod_trace_dir = make_mod_trace_dir(algorithm, mod_INQConv2d.__class__.__name__)
    store_trace(G_INQConv2d, mod_trace_dir)
    draw_graph(G_INQConv2d, dir=mod_trace_dir)

    # mod_INQConv3d = qa.inq.INQConv3d(_n_input_channels, _n_output_channels, kernel_size=_kernel_size, stride=_stride, padding=_padding, bias=False)
    # mod_INQConv3d.eval()
    # G_INQConv3d = trace_pytorch_module(mod_INQConv3d, torch.ones((_batch_size, _n_input_channels, _dim1, _dim2, _dim3)))
    # mod_trace_dir = make_mod_trace_dir(algorithm, mod_INQConv3d.__class__.__name__)
    # store_trace(G_INQConv3d, mod_trace_dir)
    # draw_graph(G_INQConv3d, dir=mod_trace_dir)


if __name__ == '__main__':

    if not os.path.isdir(__TRACES_DIR__):
        os.mkdir(__TRACES_DIR__)

    pytorch()
    quantlib()
