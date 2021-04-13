import torch
import torch.onnx.utils
import re
from packaging import version

import networkx as nx
import itertools


__all__ = [
    'ONNXGraph',
]


__KERNEL_PARTITION__ = 0
__MEMORY_PARTITION__ = 1
__CONTXT_PARTITION__ = 2


# necessary for PyTorch >= 1.4 (see https://github.com/pytorch/pytorch/issues/33463#issuecomment-606399944)
class scope_name_workaround(object):

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


class QuantLabNode(object):

    def __init__(self, obj):
        self.nobj = obj


class ONNXNode(QuantLabNode):

    def __init__(self, obj):
        super(ONNXNode, self).__init__(obj)
        self.nname = '/'.join([self.nscope, self.ntype])

    @staticmethod
    def onnx_scope_2_pytorch_scope(scope_name):
        module_name_parts = re.findall('\[.*?\]', scope_name)
        return '.'.join([mn[1:-1] for mn in module_name_parts])

    @property
    def nscope(self):
        if isinstance(self.nobj, torch._C.Node):
            nscope = ONNXNode.onnx_scope_2_pytorch_scope(self.nobj.scopeName())
        elif isinstance(self.nobj, torch._C.Value):
            nscope = self.nobj.debugName()
        return nscope

    @property
    def ntype(self):
        if isinstance(self.nobj, torch._C.Node):
            ntype = self.nobj.kind()
        elif isinstance(self.nobj, torch._C.Value):
            ntype = '*'  # data nodes are untyped ('onnx::Tensor'?)
        return ntype


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
        opnodes_dict = dict()
        datanodes_dict = dict()
        arcs = list()

        node_id_format = '{:06d}'

        datanode_id_gen = itertools.count()
        datanodes_2_id_dict = dict()  # data nodes will be discovered: I do not know in advance who they are

        for i_op, opnode in enumerate(self.jit_graph.nodes()):

            # populate kernel partition of the computational graph
            opnode_id = 'O' + node_id_format.format(i_op)
            opnodes_dict[opnode_id] = ONNXNode(opnode)  # get_opnode_attributes(opnode)

            # populate memory partition of the computational graph
            # I might encouter the same data node ('torch._C.Value') again in
            # other iterations of the loop on op nodes. I am trusting the fact
            # that the object will have the same `debugName`; seems
            # reasonable, if nobody (or nothing) touches the object in-between
            # iterations.
            for in_datanode in opnode.inputs():
                datanode_name = in_datanode.debugName()
                try:  # the data node has already been discovered
                    datanode_id = datanodes_2_id_dict[datanode_name]
                except KeyError:
                    datanode_id = 'D' + node_id_format.format(next(datanode_id_gen))
                    datanodes_2_id_dict[datanode_name] = datanode_id
                    datanodes_dict[datanode_id] = ONNXNode(in_datanode)  # get_datanode_attributes(in_datanode)
                arcs.append((datanode_id, opnode_id))

            for out_datanode in opnode.outputs():
                datanode_name = out_datanode.debugName()
                try:  # the data node has already been discovered
                    datanode_id = datanodes_2_id_dict[datanode_name]
                except KeyError:
                    datanode_id = 'D' + node_id_format.format(next(datanode_id_gen))
                    datanodes_2_id_dict[datanode_name] = datanode_id
                    datanodes_dict[datanode_id] = ONNXNode(out_datanode)  # get_datanode_attributes(out_datanode)
                arcs.append((opnode_id, datanode_id))

        self.nx_graph = nx.DiGraph()
        self.nx_graph.add_nodes_from(list(opnodes_dict.keys()), bipartite=__KERNEL_PARTITION__)
        self.nx_graph.add_nodes_from(list(datanodes_dict.keys()), bipartite=__MEMORY_PARTITION__)
        self.nx_graph.add_edges_from(arcs)

        self.nodes_dict = {**opnodes_dict, **datanodes_dict}

        nx.set_node_attributes(self.nx_graph, {k: v.nscope for k, v in self.nodes_dict.items()}, 'scope')
        nx.set_node_attributes(self.nx_graph, {k: v.ntype for k, v in self.nodes_dict.items()}, 'type')
        nx.set_node_attributes(self.nx_graph, {k: v.nname for k, v in self.nodes_dict.items()}, 'name')

    def rescope_opnodes(self, algorithms=None):

        from .trace import load_traces_library
        from .grrules import ManualRescopeRule, AutoRescopeRule

        libtraces = load_traces_library(algorithms=algorithms)
        librules = dict()
        for mod_name, G in libtraces.items():
            L = G
            VK = {n for n in L.nodes if L.nodes[n]['partition'] == __CONTXT_PARTITION__}
            K = L.subgraph(VK)
            if mod_name == 'ViewFlattenNd':
                librules[mod_name] = ManualRescopeRule(L, K)
            else:
                librules[mod_name] = AutoRescopeRule(L, K)

        self.history = History()
        self.history.push(self.nx_graph)
        for mod_name, rho in librules.items():
            if mod_name == 'ViewFlattenNd':
                print("Applying ManualRescope GRR for modules of class {}...".format(mod_name))
                for g in rho.seek(self.nx_graph):
                    self.nx_graph = rho.apply(self.nx_graph, g, mod_name)
                    self.history.push((rho, g, self.nx_graph))
            else:
                print("Applying AutoRescope GRR for modules of class {}...".format(mod_name))
                for g in rho.seek(self.nx_graph):
                    self.nx_graph = rho.apply(self.nx_graph, g)
                    self.history.push((rho, g, self.nx_graph))


# @staticmethod
# def get_opnode_name(opnode, i_opnode):
#     opscope = Morpher.onnx_scope_2_pytorch_scope(opnode.scopeName())
#     opnode = opnode.kind()
#     return '/'.join([opscope, opnode_name])
#
# @staticmethod
# def get_opscope_from_opnode(opnode_name):
#     return opnode_name.split('/', 1)[0]
#
# @staticmethod
# def get_opgraph(G):
#     kernel_partition = nx.bipartite.sets(G)[__KERNEL_PARTITION__]
#     opgraph = nx.bipartite.projected_graph(G, kernel_partition)
#     return opgraph
#
# @staticmethod
# def get_template(G, in_nodes, out_nodes, include_interface=False):
#
#     if not isinstance(in_nodes, list):
#         in_nodes = [in_nodes]
#     if not isinstance(out_nodes, list):
#         out_nodes = [out_nodes]
#
#     before_in_nodes = set()
#     for node_in in in_nodes:
#         before_in_nodes = before_in_nodes | set(nx.ancestors(G, node_in))
#         # TODO: should we ensure that an input node vI1 is not amongst the ancestors of another input node vI2?
#     in_nodes = set(in_nodes)
#
#     before_out_nodes = set()
#     for node_out in out_nodes:
#         before_out_nodes = before_out_nodes | set(nx.ancestors(G, node_out))
#     out_nodes = set(out_nodes)
#
#     assert before_out_nodes.issuperset(in_nodes | before_in_nodes)
#
#     VT = before_out_nodes.difference(in_nodes | before_in_nodes)
#     if include_interface:
#         VT = out_nodes | VT | in_nodes
#
#     T = G.subgraph(VT)
#     T = nx.DiGraph(copy.copy(T))  # if I need to edit the template, it should not be a "view" (https://networkx.org/documentation/stable/reference/classes/index.html#module-networkx.classes.graphviews)
#
#     return T
#




# def rescope_opnodes(self):
#
#     def get_subgraph_scope(G, morphism):
#         partition = nx.get_node_attributes(G, 'bipartite')
#         node_names = [k for k in morphism.keys() if partition[k] == __KERNEL_PARTITION__]
#
#         scope = set(Morpher.get_opscope_from_opnode(n) for n in node_names if Morpher.get_opscope_from_opnode(n) != '')
#         assert len(scope) == 1
#
#         return next(iter(scope))  # extract the only element from the set
#
#     G = self.computational_graph
#
#     INQtemplate = Morpher.get_template(G, '174', '201')
#     STEtemplate = Morpher.get_template(G, '150', '174')
#     AvgPooltemplate = Morpher.get_template(G, '950', '959')
#
#     INQmorphisms = Morpher.get_morphisms(G, INQtemplate)
#     STEmorphisms = Morpher.get_morphisms(G, STEtemplate)
#     AvgPoolmorphisms = Morpher.get_morphisms(G, AvgPooltemplate)
#
#     for morphisms in [INQmorphisms, STEmorphisms, AvgPoolmorphisms]:
#         for g in morphisms:
#             scope = get_subgraph_scope(G, g)
#             mapping = {n: ''.join([scope, n]) if Morpher.get_opscope_from_opnode(n) == '' else n for n in g.keys()}
#             nx.relabel_nodes(G, mapping, copy=False)
#
# @staticmethod
# def find_internal_datanodes(G):
#
#     def is_datanode_internal(G, datanode, opnode):
#         neighbours = set(G.predecessors(datanode)) | set(G.successors(datanode))
#         others = neighbours.difference(set([opnode]))
#         return len(others) == 0
#
#     kernel_partition = set(nx.bipartite.sets(G)[__KERNEL_PARTITION__])
#     memory_partition = set(nx.bipartite.sets(G)[__MEMORY_PARTITION__])
#
#     for opnode in kernel_partition:
#
#         pred = set(G.predecessors(opnode))
#         succ = set(G.successors(opnode))
#         assert (pred | succ).issubset(memory_partition)
#
#         for datanode in pred & succ:
#             if is_datanode_internal(G, datanode, opnode):
#                 try:
#                     internal_datanodes.append(datanode)
#                 except UnboundLocalError:
#                     internal_datanodes = [datanode, ]
#
#     return internal_datanodes
#
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
# @staticmethod
# # this function is recursive: beware pitfalls related to inheritance! (https://stackoverflow.com/questions/13183501/staticmethod-and-recursion)
# def get_pytorch_module_by_name(module, target_name):
#
#     for name, child in module.named_children():
#         if name == target_name:
#             return child
#         elif name == target_name.split('.', 1)[0]:
#             return Morpher.get_pytorch_module_by_name(child, target_name.split('.', 1)[-1])
#
#     return module
#
# def get_pytorch_graph(self):
#
#     opnode_2_opscope = {on: Morpher.get_opscope_from_opnode(on) for on in nx.bipartite.sets(self.computational_graph)[__KERNEL_PARTITION__]}
#     datanode_2_datascope = {dn: dn for dn in nx.bipartite.sets(self.computational_graph)[__MEMORY_PARTITION__]}
#
#     P = Morpher.get_view(self.computational_graph, opnode_2_opscope, datanode_2_datascope)
#
#     kernel_partition = nx.bipartite.sets(P)[__KERNEL_PARTITION__]
#     node_2_pytorch = {n: Morpher.get_pytorch_module_by_name(self.net, n) if n in kernel_partition else 'data' for n in P.nodes}
#     node_2_label = {k: 'data' if v == 'data' else v.__class__.__name__ for k, v in node_2_pytorch.items()}
#     nx.set_node_attributes(P, node_2_pytorch, 'pytorch')
#     nx.set_node_attributes(P, node_2_label, 'label')
#
#     return P
#
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
            force = confirmation in ('y', 'Y')

        if force:
            self._undo = list()
            self._redo = list()


# 1. label the graph nodes
# 2. define a rule which is based on the defined labelling (morphisms preserve labelling)
# 3. 'discover' possible application points for the rules
# 4. 'filter' the sequence of application points (NOT automatic)
# 5. 'apply' the rule to the filtered sequence of application points
#     - the pair (rule, application_points) is called a 'transform'
# 6. 'generate_code' for the transformed graph
# 7. 'import_network' from the transformed graph's file

# [COMMENT 1] Steps 1 and 2 are usually designed in reversed order:
#   - the user first thinks to the rule
#   - then decides which "pieces" should be in the label
# which "ingredients" did I use in the past to generate these labels? (my personal "database/record" of use cases)
#   - ONNX op type
#   - node name
