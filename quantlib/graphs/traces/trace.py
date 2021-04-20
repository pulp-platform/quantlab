import networkx as nx
import os

import quantlib.graphs.graphs
import quantlib.graphs.utils


__TRACES_LIBRARY__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.libtraces')


def trace_module(library, algorithm, mod, dummy_input):

    # trace graph
    mod.eval()
    onnxgraph = quantlib.graphs.graphs.ONNXGraph(mod, dummy_input)
    G = onnxgraph.nx_graph

    # locate interface nodes
    node_2_partition = nx.get_node_attributes(G, 'bipartite')
    for n in {n for n, onnxnode in onnxgraph.nodes_dict.items() if onnxnode.nobj in set(onnxgraph.jit_graph.inputs()) | set(onnxgraph.jit_graph.outputs())}:
        node_2_partition[n] = quantlib.graphs.graphs.__CONTXT_PARTITION__
    nx.set_node_attributes(G, node_2_partition, 'partition')

    # store traces and graph picture
    trace_dir = os.path.join(__TRACES_LIBRARY__, library, algorithm, mod.__class__.__name__)
    if not os.path.isdir(trace_dir):
        os.makedirs(trace_dir, exist_ok=True)
    nx.write_gpickle(G, os.path.join(trace_dir, 'networkx'))
    quantlib.graphs.utils.draw_graph(G, trace_dir, 'graphviz')


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

def trace_pytorch_modules():

    import torch
    import torch.nn as nn

    library = 'PyTorch'

    #####################
    ## AdaptiveAvgPool ##
    #####################
    algorithm = 'AdaptiveAvgPool'

    mod_AdaptiveAvgPool1d = nn.AdaptiveAvgPool1d((int(_dim1 / 4)))
    dummy_input = torch.ones(_batch_size, _n_input_channels, _dim1)
    trace_module(library, algorithm, mod_AdaptiveAvgPool1d, dummy_input)

    mod_AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((int(_dim1 / 4), int(_dim2 / 4)))
    dummy_input = torch.ones(_batch_size, _n_input_channels, _dim1, _dim2)
    trace_module(library, algorithm, mod_AdaptiveAvgPool2d, dummy_input)

    mod_AdaptiveAvgPool3d = nn.AdaptiveAvgPool3d((int(_dim1 / 4), int(_dim2 / 4), int(_dim3 / 4)))
    dummy_input = torch.ones(_batch_size, _n_input_channels, _dim1, _dim2, _dim3)
    trace_module(library, algorithm, mod_AdaptiveAvgPool3d, dummy_input)

    ################
    ## torch.view ##
    ################
    algorithm = 'ViewFlatten'

    mod_ViewFlattenNd = quantlib.graphs.graphs.ViewFlattenNd()
    dummy_input = torch.ones(_batch_size, _n_input_channels, _dim1, _dim2, _dim3)
    trace_module(library, algorithm, mod_ViewFlattenNd, dummy_input)


##############################
## QUANTLIB MODULES TRACING ##
##############################

def trace_quantlib_modules():

    import torch
    import quantlib.algorithms as qa

    library = 'QuantLab'

    #########
    ## STE ##
    #########
    algorithm = 'STE'

    mod_STEActivation = qa.ste.STEActivation()
    dummy_input = torch.ones((_batch_size, _n_input_channels))
    trace_module(library, algorithm, mod_STEActivation, dummy_input)

    #########
    ## INQ ##
    #########
    algorithm = 'INQ'

    mod_INQConv1d = qa.inq.INQConv1d(_n_input_channels, _n_output_channels, kernel_size=_kernel_size, stride=_stride, padding=_padding, bias=False)
    dummy_inpyut = torch.ones((_batch_size, _n_input_channels, _dim1))
    trace_module(library, algorithm, mod_INQConv1d, dummy_inpyut)

    mod_INQConv2d = qa.inq.INQConv2d(_n_input_channels, _n_output_channels, kernel_size=_kernel_size, stride=_stride, padding=_padding, bias=False)
    dummy_input = torch.ones((_batch_size, _n_input_channels, _dim1, _dim2))
    trace_module(library, algorithm, mod_INQConv2d, dummy_input)

    # mod_INQConv3d = qa.inq.INQConv3d(_n_input_channels, _n_output_channels, kernel_size=_kernel_size, stride=_stride, padding=_padding, bias=False)
    # dummy_input = torch.ones((_batch_size, _n_input_channels, _dim1, _dim2, _dim3))
    # trace_module(library, algorithm, mod_INQConv3d, dummy_input)


if __name__ == '__main__':

    if not os.path.isdir(__TRACES_LIBRARY__):
        os.mkdir(__TRACES_LIBRARY__)

    trace_pytorch_modules()
    trace_quantlib_modules()
