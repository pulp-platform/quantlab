import torch
import torch.onnx.utils
import re
from packaging import version
import networkx as nx
import itertools

import quantlib.graphs.trace
import quantlib.graphs.grrules
import quantlib.graphs.utils


__all__ = [
    'ONNXGraph',
    'PyTorchGraph',
]


__KERNEL_PARTITION__ = 0
__MEMORY_PARTITION__ = 1
__CONTXT_PARTITION__ = 2


class History(object):

    def __init__(self):
        self._undo = list()
        self._redo = list()

    def show(self):
        print("-- History --")
        for i, item in enumerate(self._undo):
            print(i, item)

    def push(self, item):
        self._undo.append(item)
        self._redo = list()

    def undo(self, n=1):
        for i in range(0, n):
            try:
                self._redo.append(self._undo.pop())
            except IndexError:
                print("Tried to undo {} steps, but history contained just {}. 'Undo' stack has been cleared.".format(n, i))
                break

    def redo(self, n=1):
        for i in range(0, n):
            try:
                self._undo.append(self._redo.pop())
            except IndexError:
                print("Tried to redo {} steps, but history contained just {}. 'Redo' stack has been cleared.".format(n, i))
                break

    def clear(self, force=False):

        if not force:
            confirmation = input("This action is not reversible. Are you sure that you want to delete all the history? [y/N]")
            force = confirmation.lower() == 'y'

        if force:
            self._undo.clear()
            self._redo.clear()


# 1. label graph nodes (node label is usually computed as the aggregation of 1.partition and 2.type, but see COMMENT below)
# 2. define a graph rewriting rule (GRR)
# 3. 'discover' possible application points for the rules
# 4. 'filter' the sequence of application points (possibly NOT automatic)
# 5. 'apply' the rule to the filtered sequence of application points
#     - each pair (rule, application_point) is called a 'transform', and the resulting graph is called a 'derivation'
# 6. 'generate_code' for the transformed graph
# 7. 'import_network' from the transformed graph's file

# [COMMENT] Steps 1 and 2 are usually designed in reversed order:
#   - the user first thinks to the rule
#   - then decides which "pieces" should be in the label
# which "ingredients" did I use in the past to generate these labels? (my personal "database/record" of use cases)
#   - ONNX op type
#   - node scope


class QuantLabNode(object):

    def __init__(self, obj):
        self.nobj = obj


class scope_name_workaround(object):
    # necessary for PyTorch >= 1.4 (see https://github.com/pytorch/pytorch/issues/33463#issuecomment-606399944)

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


class ONNXNode(QuantLabNode):

    def __init__(self, obj):
        super(ONNXNode, self).__init__(obj)

    @staticmethod
    def onnx_scope_2_pytorch_scope(onnx_scope):
        module_name_parts = re.findall('\[.*?\]', onnx_scope)
        pytorch_scope = '.'.join([mn[1:-1] for mn in module_name_parts])
        return pytorch_scope

    @property
    def ntype(self):
        if isinstance(self.nobj, torch._C.Node):
            ntype = self.nobj.kind()
        elif isinstance(self.nobj, torch._C.Value):
            ntype = '*'  # data nodes are untyped ('onnx::Tensor'?)
        return ntype

    @property
    def nscope(self):
        if isinstance(self.nobj, torch._C.Node):
            nscope = ONNXNode.onnx_scope_2_pytorch_scope(self.nobj.scopeName())
        elif isinstance(self.nobj, torch._C.Value):
            nscope = self.nobj.debugName()
        return nscope


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

        node_id_format = '{:06d}'
        datanode_2_onnx_id = dict()  # data nodes will be discovered via tracing (I do not know in advance who they are)
        datanode_id_gen = itertools.count()

        for i_op, opnode in enumerate(self.jit_graph.nodes()):

            # populate kernel partition of the computational graph
            opnode_id = 'O' + node_id_format.format(i_op)
            opnodes_dict[opnode_id] = ONNXNode(opnode)

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
                    datanode_id = 'D' + node_id_format.format(next(datanode_id_gen))
                    datanode_2_onnx_id[datanode_name] = datanode_id
                    datanodes_dict[datanode_id] = ONNXNode(in_datanode)
                arcs.append((datanode_id, opnode_id))

            for out_datanode in opnode.outputs():
                datanode_name = out_datanode.debugName()
                try:  # the data node has already been discovered
                    datanode_id = datanode_2_onnx_id[datanode_name]
                except KeyError:
                    datanode_id = 'D' + node_id_format.format(next(datanode_id_gen))
                    datanode_2_onnx_id[datanode_name] = datanode_id
                    datanodes_dict[datanode_id] = ONNXNode(out_datanode)  # get_datanode_attributes(out_datanode)
                arcs.append((opnode_id, datanode_id))

        self.nx_graph = nx.DiGraph()
        self.nx_graph.add_nodes_from(set(opnodes_dict.keys()), bipartite=__KERNEL_PARTITION__)
        self.nx_graph.add_nodes_from(set(datanodes_dict.keys()), bipartite=__MEMORY_PARTITION__)
        self.nx_graph.add_edges_from(arcs)

        self.nodes_dict = {**opnodes_dict, **datanodes_dict}

        nx.set_node_attributes(self.nx_graph, {k: v.ntype for k, v in self.nodes_dict.items()}, 'type')
        nx.set_node_attributes(self.nx_graph, {k: v.nscope for k, v in self.nodes_dict.items()}, 'scope')

        # editing tracker
        self.history = History()
        self.history.push((None, None, self.nx_graph))

    def rescope_opnodes(self, modules=None):

        libtraces = quantlib.graphs.trace.load_traces_library(modules=modules)
        librules = {}
        for mod_name, (L, K) in libtraces.items():
            if mod_name == 'ViewFlattenNd':
                librules[mod_name] = quantlib.graphs.grrules.ManualRescopeRule(L, K)
            else:
                librules[mod_name] = quantlib.graphs.grrules.AutoRescopeRule(L, K)

        for mod_name, rho in librules.items():
            if mod_name == 'ViewFlattenNd':
                print("Applying ManualRescope GRR for `nn.Module`s of class {}...".format(mod_name))
                for i, g in enumerate(rho.seek(self.nx_graph)):
                    self.nx_graph = rho.apply(self.nx_graph, g, '.'.join([mod_name, str(i)]))
                    self.history.push((rho, g, self.nx_graph))
            else:
                print("Applying AutoRescope GRR for `nn.Module`s of class {}...".format(mod_name))
                for g in rho.seek(self.nx_graph):
                    self.nx_graph = rho.apply(self.nx_graph, g)
                    self.history.push((rho, g, self.nx_graph))


class PyTorchNode(QuantLabNode):

    def __init__(self, obj):
        super(PyTorchNode, self).__init__(obj)

    @property
    def ntype(self):
        if isinstance(self.nobj, torch.nn.Module):
            ntype = self.nobj.__class__.__name__
        elif isinstance(self.nobj, torch._C.Value):
            ntype = '*'  # data nodes are untyped ('onnx::Tensor'?)
        return ntype

    @property
    def nscope(self):
        if isinstance(self.nobj, torch.nn.Module):
            nscope = ''  # the scope of `nn.Module`s usually depends on the "view" that the network's coder had of it at implementation time; we leave op nodes unscoped
        elif isinstance(self.nobj, torch._C.Value):
            nscope = self.nobj.debugName()
        return nscope


class PyTorchGraph(object):

    def __init__(self, net, onnxgraph):

        # map ONNX graph to PyTorch graph
        G = onnxgraph.nx_graph
        assert '' not in {G.nodes[n]['scope'] for n in G.nodes}, "Argument graph {} has unscoped nodes.".format(G)

        g = nx.get_node_attributes(G, 'scope')
        opnodes   = {scope for n, scope in g.items() if G.nodes[n]['bipartite'] == __KERNEL_PARTITION__}
        datanodes = {scope for n, scope in g.items() if G.nodes[n]['bipartite'] == __MEMORY_PARTITION__}
        arcs      = {arc for arc in map(lambda a: (g[a[0]], g[a[-1]]), G.edges)}

        self.nx_graph = nx.DiGraph()
        self.nx_graph.add_nodes_from(opnodes, bipartite=__KERNEL_PARTITION__)
        self.nx_graph.add_nodes_from(datanodes, bipartite=__MEMORY_PARTITION__)
        self.nx_graph.add_edges_from(arcs)

        # remove data nodes which are used internaly to a PyTorch `nn.Module` (i.e., as "working memory");
        # beware: this operation is not reversible!
        self.nx_graph.remove_nodes_from(PyTorchGraph.find_internal_datanodes(self.nx_graph))

        # reassign IDs to nodes based on their topological sorting (I assume the graph is a DAG)
        node_id_format = '{:06d}'
        opnode_id_gen   = itertools.count()
        datanode_id_gen = itertools.count()

        onnx_scope_2_pytorch_id = {}
        from networkx.algorithms import dag
        for n in dag.topological_sort(self.nx_graph):
            if self.nx_graph.nodes[n]['bipartite'] == __KERNEL_PARTITION__:
                node_id = 'O' + node_id_format.format(next(opnode_id_gen))
            elif self.nx_graph.nodes[n]['bipartite'] == __MEMORY_PARTITION__:
                node_id = 'D' + node_id_format.format(next(datanode_id_gen))
            onnx_scope_2_pytorch_id[n] = node_id

        nx.relabel_nodes(self.nx_graph, onnx_scope_2_pytorch_id, copy=False)
        self.pytorch_id_2_onnx_scope = {v: k for k, v in onnx_scope_2_pytorch_id.items()}

        # assign type and scope attributes to PyTorch graph nodes
        opnodes_dict   = {}
        datanodes_dict = {}

        # populate kernel partition of the computational graph
        for n in {n for n in self.nx_graph.nodes if self.nx_graph.nodes[n]['bipartite'] == __KERNEL_PARTITION__}:
            onnx_scope = self.pytorch_id_2_onnx_scope[n]
            if 'ViewFlattenNd' in onnx_scope:
                obj = quantlib.graphs.utils.ViewFlattenNd()
            else:
                obj = PyTorchGraph.get_pytorch_module_by_name(net, onnx_scope)
            opnodes_dict[n] = PyTorchNode(obj)

        # populate memory partition of the computational graph
        onnx_scope_2_onnx_id = {v: k for k, v in nx.get_node_attributes(G, 'scope').items()}
        for n in {n for n in self.nx_graph.nodes if self.nx_graph.nodes[n]['bipartite'] == __MEMORY_PARTITION__}:
            onnx_scope = self.pytorch_id_2_onnx_scope[n]
            obj = onnxgraph.nodes_dict[onnx_scope_2_onnx_id[onnx_scope]].nobj
            datanodes_dict[n] = PyTorchNode(obj)

        self.nodes_dict = {**opnodes_dict, **datanodes_dict}

        nx.set_node_attributes(self.nx_graph, {k: v.ntype for k, v in self.nodes_dict.items()}, 'type')
        nx.set_node_attributes(self.nx_graph, {k: v.nscope for k, v in self.nodes_dict.items()}, 'scope')

        # editing tracker
        self.history = History()
        self.history.push((None, None, self.nx_graph))

    @staticmethod
    def find_internal_datanodes(G):

        opnodes   = {n for n in G.nodes if G.nodes[n]['bipartite'] == __KERNEL_PARTITION__}
        datanodes = {n for n in G.nodes if G.nodes[n]['bipartite'] == __MEMORY_PARTITION__}

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

# @staticmethod
# def get_opgraph(G):
#     kernel_partition = nx.bipartite.sets(G)[__KERNEL_PARTITION__]
#     opgraph = nx.bipartite.projected_graph(G, kernel_partition)
#     return opgraph

# @staticmethod
# def get_view(G, opnode_2_opscope, datanode_2_datascope):
#     """Morph a low-level view into a higher-level view of the computational graph."""
#
#     node_2_scope = {**opnode_2_opscope, **datanode_2_datascope}
#     arcs = list(map(lambda arc: (node_2_scope[arc[0]], node_2_scope[arc[1]]), G.edges))
#
#     F = nx.DiGraph()  # the higher-level view is a brand-new graph object
#     F.add_nodes_from(set(opnode_2_opscope.values()), bipartite=__KERNEL_PARTITION__)
#     F.add_nodes_from(set(datanode_2_datascope.values()), bipartite=__MEMORY_PARTITION__)
#     F.add_edges_from(arcs)
#
#     internal_datanodes = Morpher.find_internal_datanodes(F)
#     F.remove_nodes_from(internal_datanodes)
#
#     return F
#



# def add_ste_tunnels(G):
#
#     H = copy.deepcopy(G)
#
#     import torch.nn as nn
#     from quantlib.algorithms.ste import STEActivation
#
#     ste_modules = [n[0] for n in H.nodes.data() if isinstance(n[1]['pytorch'], STEActivation)]
#
#     def add_tunnel(H, in_node_name, out_node_name, i):
#
#         tunnel_node_name = '.'.join([in_node_name, 'tunnel', str(i)])
#         tunnel_node_data = copy.deepcopy(H.nodes[in_node_name])
#         tunnel_node_data['pytorch'] = nn.Identity()
#         tunnel_node_data['label'] = 'tunnel'
#
#         copy_node_name = '.'.join([in_node_name, 'copy', str(i)])
#         copy_node_data = copy.deepcopy(H.nodes[in_node_name])
#
#         H.add_node(tunnel_node_name, **tunnel_node_data)
#         H.add_node(copy_node_name, **copy_node_data)
#
#         H.add_edge(in_node_name, tunnel_node_name)
#         H.add_edge(tunnel_node_name, copy_node_name)
#         H.add_edge(copy_node_name, out_node_name)
#
#         H.remove_edge(in_node_name, out_node_name)
#
#     for node_name in ste_modules:
#         next_node_names = list(H.successors(node_name))
#         for i, next_name in enumerate(next_node_names):
#             add_tunnel(H, node_name, next_name, i)
#
#     return H
#
#
# def add_linear_tunnels(G):
#
#     H = copy.deepcopy(G)
#
#     import torch.nn as nn
#
#     linear_modules = [n[0] for n in H.nodes.data() if isinstance(n[1]['pytorch'], nn.Linear)]
#
#     def add_tunnel(H, in_node_name, out_node_name, i):
#
#         tunnel_node_name = '.'.join([in_node_name, 'tunnel', str(i)])
#         tunnel_node_data = copy.deepcopy(H.nodes[in_node_name])
#         tunnel_node_data['pytorch'] = nn.Identity()
#         tunnel_node_data['label'] = 'tunnel'
#
#         H.add_node(tunnel_node_name, **tunnel_node_data)
#
#         H.add_edge(in_node_name, tunnel_node_name)
#         H.add_edge(tunnel_node_name, out_node_name)
#
#         H.remove_edge(in_node_name, out_node_name)
#
#     for node_name in linear_modules:
#         prec_node_names = list(H.predecessors(node_name))
#         for i, prec_name in enumerate(prec_node_names):
#             add_tunnel(H, prec_name, node_name, i)
#
#     return H
#
#
# def add_output_tunnel(G, last_node_name):
#
#     H = copy.deepcopy(G)
#
#     import torch.nn as nn
#
#     tunnel_node_name = '__output_tunnel'
#     tunnel_node_data = copy.deepcopy(H.nodes[last_node_name])
#     tunnel_node_data['pytorch'] = nn.Identity()
#     tunnel_node_data['label'] = 'tunnel'
#
#     H.add_node(tunnel_node_name, **tunnel_node_data)
#     H.add_edge(last_node_name, tunnel_node_name)
#
#     return H
#
#
# def remove_tunnels(G):
#
#     H = copy.deepcopy(G)
#
#     import torch.nn as nn
#     import itertools
#
#     tunnel_modules = [n[0] for n in H.nodes.data() if 'tunnel' in n[0] and isinstance(n[1]['pytorch'], nn.Identity)]
#     for node in tunnel_modules:
#         for (in_node, out_node) in itertools.product(H.predecessors(node), H.successors(node)):
#             H.add_edge(in_node, out_node)
#         H.remove_node(node)
#
#     return H
#
#
# # exec(open('converter_twn.py').read())
# #
# # from quantlib.graphs.morph.analyse import Morpher, add_ste_tunnels, remove_tunnels
# #
# # netmorpher.rescope_opnodes()
# # P = netmorpher.get_opgraph(netmorpher.get_pytorch_graph())
# # Q = add_ste_tunnels(P)
# # R = remove_tunnels(Q)
# #
# # import networkx as nx
# #
# # H2Dtemplate = Q.subgraph(nx.ancestors(Q, 'features.0.ste.tunnel.0'))
# # D2Dtemplate = Q.subgraph(set(nx.descendants(Q, 'features.0.ste.tunnel.0')) & set(nx.ancestors(Q, 'features.4.ste.tunnel.0')))
# # [D2Dtemplate.nodes[n]['pytorch'] for n in nx.algorithms.dag.topological_sort(D2Dtemplate)]
# # # [STEActivation(), INQConv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(inplace=True), MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), STEActivation()]
# # D2Dtemplate2 = Q.subgraph(set(nx.descendants(Q, 'features.4.ste.tunnel.0')) & set(nx.ancestors(Q, 'features.7.ste.tunnel.0')))
# # [D2Dtemplate2.nodes[n]['pytorch'] for n in nx.algorithms.dag.topological_sort(D2Dtemplate2)]
# # # [STEActivation(), INQConv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(inplace=True), STEActivation()]
# # H2Dmorphisms = Morpher.get_morphisms(Q, H2Dtemplate, 'pytorch')
# # len(H2Dmorphisms)
#
#

# # import networkx as nx
# # G = nx.DiGraph()
# # G.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e'), ('e', 'f')])
# # nx.set_node_attributes(G, 0, 'bipartite')
# # nx.set_node_attributes(G, 'same', 'label')
# # G.nodes['a']['label'] = 'diff'
# # G.nodes['f']['label'] = 'diff'
# # import copy
# # P = copy.deepcopy(G)
# # from quantlib.graphs.morph.morpher import ScopeRule
# # rho = ScopeRule(G, set(['a', 'f']))
# # g = rho.discover(P)[0]
# # P = rho.apply(P, g, 'scope')
