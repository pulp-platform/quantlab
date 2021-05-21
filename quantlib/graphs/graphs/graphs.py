from packaging import version
import torch.onnx.utils
import torch
import itertools
import networkx as nx
rom networkx.algorithms import dag

import quantlib.graphs.graphs


class scope_name_workaround(object):
    # this is a necessary "context manager" object for PyTorch >= 1.4
    # (see https://github.com/pytorch/pytorch/issues/33463#issuecomment-606399944)

    def __init__(self):
        self.backup = None  # store pointer to the definition of the "native" '_slow_forward'

    def __enter__(self):

        def _tracing_name(self_, tracing_state):

            if not tracing_state._traced_module_stack:
                return None

            module = tracing_state._traced_module_stack[-1]
            for name, child in module.named_children():
                if child is self_:
                    return name

            return None

        def _slow_forward(self_, *input, **kwargs):

            tracing_state = torch._C._get_tracing_state()

            if not tracing_state or isinstance(self_.forward, torch._C.ScriptMethod):
                return self_.forward(*input, **kwargs)  # no need to wrap and trace

            if not hasattr(tracing_state, '_traced_module_stack'):
                tracing_state._traced_module_stack = []

            name = _tracing_name(self_, tracing_state)
            if name:
                tracing_state.push_scope('{}[{}]'.format(self_._get_name(), name))
            else:
                tracing_state.push_scope(self_._get_name())
            tracing_state._traced_module_stack.append(self_)

            try:
                result = self_.forward(*input, **kwargs)
            finally:
                tracing_state.pop_scope()
                tracing_state._traced_module_stack.pop()

            return result

        self.backup = torch.nn.Module._slow_forward
        setattr(torch.nn.Module, '_slow_forward', _slow_forward)  # replace '_slow_forward' with the version defined by this context manager

    def __exit__(self, type, value, tb):
        setattr(torch.nn.Module, '_slow_forward', self.backup)  # restore "native" '_slow_forward' method


class ONNXGraph(object):

    def __init__(self, net, dummy_input):

        if not isinstance(dummy_input, tuple):
            dummy_input = (dummy_input,)

        assert version.parse(torch.__version__) >= version.parse('1.4.0')
        from torch.onnx.symbolic_helper import _set_opset_version
        _set_opset_version(11)  # opset_version9 does not support `round` ONNX operator (even though docs for PyTorch 1.5.0 suggests so)

        with scope_name_workaround():
            self.jit_graph, _, _ = torch.onnx.utils._model_to_graph(net, dummy_input, propagate=True, _retain_param_name=True)

        # At this point, I have a handle on a `torch._C.Graph` object; its
        # components are `torch._C.Node` objects, which are abstractions for
        # operations; the "data pools" where operands (i.e., inputs) are read
        # from and results (i.e., outputs) are written to are `torch._C.Value`
        # objects. The definitions of these objects can be found in the file
        # "torch/csrc/jit/ir/ir.h" in PyTorch's codebase:
        #
        #     https://github.com/pytorch/pytorch .
        #
        # More specifically, look for the following definitions:
        #  - 'struct Value';
        #  - 'struct TORCH_API Node';
        #  - 'struct Graph'.
        #
        # These structures are exposed in Python via the 'pybind11' module.

        # build computational graph
        opnodes_dict   = {}
        datanodes_dict = {}
        arcs           = []

        datanode_2_onnx_id = dict()  # data nodes will be discovered via tracing (I do not know in advance who they are)
        datanode_id_gen = itertools.count()

        for i_op, opnode in enumerate(self.jit_graph.nodes()):

            # populate kernel partition of the computational graph
            opnode_id = 'O' + quantlib.graphs.graphs.__NODE_ID_FORMAT__.format(i_op)
            opnodes_dict[opnode_id] = quantlib.graphs.graphs.nodes.ONNXNode(opnode)

            # populate memory partition of the computational graph
            # I might encouter the same data node ('torch._C.Value') again in
            # other iterations of the loop on op nodes. I am trusting the fact
            # that the object will have the same `debugName`; this seems
            # reasonable, if nobody (or nothing) modifies the object
            # in-between iterations.
            for in_datanode in opnode.inputs():
                datanode_name = in_datanode.debugName()
                try:  # the data node has already been discovered
                    datanode_id = datanode_2_onnx_id[datanode_name]
                except KeyError:
                    datanode_id = 'D' + quantlib.graphs.graphs.__NODE_ID_FORMAT__.format(next(datanode_id_gen))
                    datanode_2_onnx_id[datanode_name] = datanode_id
                    datanodes_dict[datanode_id] = quantlib.graphs.graphs.nodes.ONNXNode(in_datanode)
                arcs.append((datanode_id, opnode_id))

            for out_datanode in opnode.outputs():
                datanode_name = out_datanode.debugName()
                try:  # the data node has already been discovered
                    datanode_id = datanode_2_onnx_id[datanode_name]
                except KeyError:
                    datanode_id = 'D' + quantlib.graphs.graphs.__NODE_ID_FORMAT__.format(next(datanode_id_gen))
                    datanode_2_onnx_id[datanode_name] = datanode_id
                    datanodes_dict[datanode_id] = quantlib.graphs.graphs.nodes.ONNXNode(out_datanode)  # get_datanode_attributes(out_datanode)
                arcs.append((opnode_id, datanode_id))

        self.nx_graph = nx.DiGraph()
        self.nx_graph.add_nodes_from(set(opnodes_dict.keys()), bipartite=quantlib.graphs.graphs.__KERNEL_PARTITION__)
        self.nx_graph.add_nodes_from(set(datanodes_dict.keys()), bipartite=quantlib.graphs.graphs.__MEMORY_PARTITION__)
        self.nx_graph.add_edges_from(set(arcs))

        self.nodes_dict = {**opnodes_dict, **datanodes_dict}

        nx.set_node_attributes(self.nx_graph, {k: v.ntype for k, v in self.nodes_dict.items()}, 'type')
        nx.set_node_attributes(self.nx_graph, {k: v.nscope for k, v in self.nodes_dict.items()}, 'scope')


class PyTorchGraph(object):

    def __init__(self, net, onnxgraph):

        # map ONNX graph to PyTorch graph
        G = onnxgraph.nx_graph
        assert '' not in {G.nodes[n]['scope'] for n in G.nodes}, "Argument graph {} has unscoped nodes.".format(G)

        g = nx.get_node_attributes(G, 'scope')
        opnodes   = {scope for n, scope in g.items() if G.nodes[n]['bipartite'] == quantlib.graphs.graphs.__KERNEL_PARTITION__}
        datanodes = {scope for n, scope in g.items() if G.nodes[n]['bipartite'] == quantlib.graphs.graphs.__MEMORY_PARTITION__}
        arcs      = {arc for arc in map(lambda a: (g[a[0]], g[a[-1]]), G.edges)}

        self.nx_graph = nx.DiGraph()
        self.nx_graph.add_nodes_from(opnodes, bipartite=quantlib.graphs.graphs.__KERNEL_PARTITION__)
        self.nx_graph.add_nodes_from(datanodes, bipartite=quantlib.graphs.graphs.__MEMORY_PARTITION__)
        self.nx_graph.add_edges_from(arcs)

        # remove data nodes which are used internaly to a PyTorch `nn.Module` (i.e., as "working memory");
        # beware: this operation is not reversible!
        self.nx_graph.remove_nodes_from(PyTorchGraph.find_internal_datanodes(self.nx_graph))

        # reassign IDs to nodes based on their topological sorting (I assume the graph is a DAG)
        __NODE_ID_FORMAT__ = '{:06d}'
        opnode_id_gen   = itertools.count()
        datanode_id_gen = itertools.count()

        onnx_scope_2_pytorch_id = {}
        for n in dag.topological_sort(self.nx_graph):
            if self.nx_graph.nodes[n]['bipartite'] == quantlib.graphs.graphs.nodes.__KERNEL_PARTITION__:
                node_id = 'O' + __NODE_ID_FORMAT__.format(next(opnode_id_gen))
            elif self.nx_graph.nodes[n]['bipartite'] == quantlib.graphs.graphs.__MEMORY_PARTITION__:
                node_id = 'D' + __NODE_ID_FORMAT__.format(next(datanode_id_gen))
            onnx_scope_2_pytorch_id[n] = node_id

        nx.relabel_nodes(self.nx_graph, onnx_scope_2_pytorch_id, copy=False)
        self.pytorch_id_2_onnx_scope = {v: k for k, v in onnx_scope_2_pytorch_id.items()}

        # assign type and scope attributes to PyTorch graph nodes
        opnodes_dict   = {}
        datanodes_dict = {}

        # populate kernel partition of the computational graph
        for n in {n for n in self.nx_graph if self.nx_graph.nodes[n]['bipartite'] == quantlib.graphs.graphs.__KERNEL_PARTITION__}:
            onnx_scope = self.pytorch_id_2_onnx_scope[n]
            if 'torch.view' in onnx_scope:  # TODO: mind quantlib/graphs/grrules/__init__.py:L16
                obj = quantlib.graphs.graphs.ViewFlattenNd()
            else:
                obj = PyTorchGraph.get_pytorch_module_by_name(net, onnx_scope)
            opnodes_dict[n] = quantlib.graphs.graphs.PyTorchNode(obj)

        # populate memory partition of the computational graph
        onnx_scope_2_onnx_id = {v: k for k, v in nx.get_node_attributes(G, 'scope').items()}
        for n in {n for n in self.nx_graph.nodes if self.nx_graph.nodes[n]['bipartite'] == quantlib.graphs.graphs.__MEMORY_PARTITION__}:
            onnx_scope = self.pytorch_id_2_onnx_scope[n]
            obj = onnxgraph.nodes_dict[onnx_scope_2_onnx_id[onnx_scope]].nobj
            datanodes_dict[n] = quantlib.graphs.graphs.PyTorchNode(obj)

        self.nodes_dict = {**opnodes_dict, **datanodes_dict}

        nx.set_node_attributes(self.nx_graph, {k: v.ntype for k, v in self.nodes_dict.items()}, 'type')
        nx.set_node_attributes(self.nx_graph, {k: v.nscope for k, v in self.nodes_dict.items()}, 'scope')

    @staticmethod
    def find_internal_datanodes(G):

        opnodes   = {n for n in G.nodes if G.nodes[n]['bipartite'] == quantlib.graphs.graphs.__KERNEL_PARTITION__}
        datanodes = {n for n in G if G.nodes[n]['bipartite'] == quantlib.graphs.graphs.__MEMORY_PARTITION__}

        internal_datanodes = []
        for datanode in datanodes:

            A = set(G.predecessors(datanode))
            B = set(G.successors(datanode))
            assert B.issubset(opnodes), "Data node {} has another data node as neighbour.".format(datanode)

            if A.issubset(B) and B.issubset(A) and len(A) == 1:
                internal_datanodes.append(datanode)

        return internal_datanodes

    @staticmethod
    def get_pytorch_module_by_name(module, target_name):
        # this function is recursive: beware pitfalls related to inheritance! (https://stackoverflow.com/questions/13183501/staticmethod-and-recursion)

        for name, child in module.named_children():
            if name == target_name:
                return child
            elif name == target_name.split('.', 1)[0]:
                return PyTorchGraph.get_pytorch_module_by_name(child, target_name.split('.', 1)[-1])

        return module
