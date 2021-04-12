import torch
import torch.onnx.utils
import networkx as nx
from networkx.algorithms import isomorphism

import copy
import re
from packaging import version


__all__ = [
    'ONNXGraph',
    'ScopeRule',
]


__KERNEL_PARTITION__ = 0
__MEMORY_PARTITION__ = 1


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
        self.nodeobj = obj


class ONNXNode(QuantLabNode):

    def __init__(self, obj):
        super(ONNXNode, self).__init__(obj)
        self.nodename = '/'.join([self.nodescope, self.nodetype])

    @staticmethod
    def onnx_scope_2_pytorch_scope(scope_name):
        module_name_parts = re.findall('\[.*?\]', scope_name)
        return '.'.join([mn[1:-1] for mn in module_name_parts])

    @property
    def nodescope(self):
        if isinstance(self.nodeobj, torch._C.Node):
            nodescope = ONNXNode.onnx_scope_2_pytorch_scope(self.nodeobj.scopeName())
        elif isinstance(self.nodeobj, torch._C.Value):
            nodescope = self.nodeobj.debugName()
        return nodescope

    @property
    def nodetype(self):
        if isinstance(self.nodeobj, torch._C.Node):
            nodetype = self.nodeobj.kind()
        elif isinstance(self.nodeobj, torch._C.Value):
            nodetype = '*'  # data nodes are untyped ('onnx::Tensor'?)
        return nodetype


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
        # and results (i.e., outputs) are written are `torch._C.Value`
        # objects. The definitions of these objects can be found in the file
        # "torch/csrc/jit/ir/ir.h" in PyTorch's codebase
        #
        #     (https://github.com/pytorch/pytorch)
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

        datanodes_2_id_dict = dict()  # data nodes will be discovered: I do not know in advance who they are
        def datanode_id_generator():
            id = 0
            while True:
                yield str(id)
                id += 1
        datanode_id_gen = datanode_id_generator()

        for i_op, opnode in enumerate(self.jit_graph.nodes()):

            # populate kernel partition of the computational graph
            opnode_id = 'O' + str(i_op)
            opnodes_dict[opnode_id] = ONNXNode(opnode)  # get_opnode_attributes(opnode)

            # populate memory partition of the computational graph
            # I might encouter the same data node ('torch._C.Value') again in other iterations of the loop on op odes.
            # I am trusting the fact that the object will have the same `debugName`;
            # seems reasonable, if nobody touches the object in-between iterations.
            for in_datanode in opnode.inputs():
                in_datanode_name = in_datanode.debugName()
                try:  # the data node has already been discovered
                    datanode_id = datanodes_2_id_dict[in_datanode_name]
                except KeyError:
                    datanode_id = 'D' + next(datanode_id_gen)
                    datanodes_2_id_dict[in_datanode_name] = datanode_id
                    datanodes_dict[datanode_id] = ONNXNode(in_datanode)  # get_datanode_attributes(in_datanode)
                arcs.append((datanode_id, opnode_id))

            for out_datanode in opnode.outputs():
                out_datanode_name = out_datanode.debugName()
                try:  # the data node has already been discovered
                    datanode_id = datanodes_2_id_dict[out_datanode_name]
                except KeyError:
                    datanode_id = 'D' + next(datanode_id_gen)
                    datanodes_2_id_dict[out_datanode_name] = datanode_id
                    datanodes_dict[datanode_id] = ONNXNode(out_datanode)  # get_datanode_attributes(out_datanode)
                arcs.append((opnode_id, datanode_id))

        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(list(opnodes_dict.keys()), bipartite=__KERNEL_PARTITION__)
        self.graph.add_nodes_from(list(datanodes_dict.keys()), bipartite=__MEMORY_PARTITION__)
        self.graph.add_edges_from(arcs)

        self.nodes_dict = {**opnodes_dict, **datanodes_dict}

        nx.set_node_attributes(self.graph, {k: v.nodescope for k, v in self.nodes_dict.items()}, 'scope')
        nx.set_node_attributes(self.graph, {k: v.nodetype for k, v in self.nodes_dict.items()}, 'type')
        nx.set_node_attributes(self.graph, {k: v.nodename for k, v in self.nodes_dict.items()}, 'name')

    # # assign labels to nodes (morphisms work on labelled graphs)
    # node_2_jit = nx.get_node_attributes(self.computational_graph, 'jit')
    # node_2_label = {k: v.kind() if (hasattr(v, 'kind') and callable(getattr(v, 'kind'))) else 'data' for k, v in node_2_jit.items()}
    # nx.set_node_attributes(self.computational_graph, node_2_label, 'label')
    # # Python interpreter short-circuits the evaluation of the 'and' clause:
    # # if the node attribute does not have a reference to 'kind', 'callable' won't be checked; hence, there is no risk of errors begin raised
    # # (https://docs.python.org/3/library/stdtypes.html#boolean-operations-and-or-not)
    #
    # # self.rescope_opnodes()


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

def is_morphism(G, T, H_2_T):

    # computational graphs are node-labelled graphs, where node types act as labels

    for vH, vT in H_2_T.items():

        is_same_partition = G.nodes[vH]['bipartite'] == T.nodes[vT]['bipartite']
        is_same_type = G.nodes[vH]['type'] == T.nodes[vT]['type']
        is_ok = is_same_partition and is_same_type

        if not is_ok:
            break

    return is_ok


def get_morphisms(G, T):

    # T is the "template" graph to match. In principle, morphisms need not be isomorphisms:
    # this is a restriction that I chose to simplify the work on QNNs conversion (it helps solving ambiguities).

    matcher = isomorphism.DiGraphMatcher(G, T)
    isomorphisms = list(matcher.subgraph_isomorphisms_iter())
    # candidate matchings will be "induced subgraph" isomorphisms, not "spurious" monomorphisms
    # (https://github.com/networkx/networkx/blob/master/networkx/algorithms/isomorphism/isomorphvf2.py)

    # check the second morphism condition (label consistency)
    morphisms = [g for g in isomorphisms if is_morphism(G, T, g)]

    # remove duplicate morphisms
    unique_VHs = {frozenset(g.keys()) for g in morphisms}
    VH_2_morphism = {VH: [g for g in morphisms if frozenset(g.keys()) == VH] for VH in unique_VHs}

    return [v[0] for v in VH_2_morphism.values()]
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
class Rule(object):

    def __init__(self):
        pass

    @staticmethod
    def core(H_I):
        pass

    def discover(self, G):
        pass

    def apply(self):
        pass


class ManualRescopeRule(Rule):

    def __init__(self, L, VK):

        self.L = L
        nx.relabel_nodes(self.L, {n: '/'.join(['L-term', n]) for n in set(self.L.nodes) if n not in VK}, copy=False)

        nx.relabel_nodes(self.L, {n: '/'.join(['K-term', n]) for n in set(self.L.nodes) & VK}, copy=False)
        self.K = self.L.subgraph({'/'.join(['K-term', n]) for n in VK})

        self.L_K = self.L.subgraph(set(self.L.nodes).difference({'/'.join(['K-term', n]) for n in VK}))
        K_2_L = [e for e in set(self.L.edges).difference(set(self.L_K.edges) | set(self.K.edges)) if e[0] in self.K.nodes]
        L_2_K = [e for e in set(self.L.edges).difference(set(self.L_K.edges) | set(self.K.edges)) if e[1] in self.K.nodes]

        self.R_K = nx.relabel_nodes(copy.copy(self.L_K), {n: n.replace('L-term', 'R-term') for n in self.L_K.nodes})
        L_K_2_R_K_morph = get_morphisms(self.L_K, self.R_K)
        assert len(L_K_2_R_K_morph) == 1
        L_K_2_R_K = L_K_2_R_K_morph[0]
        self.K_2_R = [(u, L_K_2_R_K[v]) for u, v in K_2_L]
        self.R_2_K = [(L_K_2_R_K[u], v) for u, v in L_2_K]

        self.S = nx.compose(self.L, self.R_K)
        self.S.add_edges_from(self.K_2_R + self.R_2_K)

    @staticmethod
    def core(H_I, new_scope):
        J_I = copy.copy(H_I)
        nx.set_node_attributes(J_I, {n: new_scope for n in J_I.nodes if (J_I.nodes[n]['bipartite'] == __KERNEL_PARTITION__)}, 'scope')
        return J_I

    def discover(self, G):
        return get_morphisms(G, self.L)

    def _apply(self, G, g, new_scope):

        H_I = G.subgraph({k for k, v in g.items() if v not in self.K.nodes})
        J_I = self.core(H_I, new_scope)

        J_I_2_R_K_morph = get_morphisms(J_I, self.R_K)
        assert len(J_I_2_R_K_morph) == 1
        J_I_2_R_K = J_I_2_R_K_morph[0]
        R_K_2_J_I = {v: k for k, v in J_I_2_R_K.items()}

        G = nx.compose(G, J_I)
        VI = {k for k, v in g.items() if v in self.K.nodes}
        for vI in VI:
            vK = g[vI]
            for e in [e for e in self.K_2_R if e[0] == vK]:
                G.add_edge(vI, R_K_2_J_I[e[1]])
            for e in [e for e in self.R_2_K if e[1] == vK]:
                G.add_edge(R_K_2_J_I[e[0]], vI)

        G.remove_nodes_from(H_I.nodes)

        return G

    def apply(self, G, gs, new_scope):
        for i, g in enumerate(gs):
            G = self._apply(G, g, new_scope + '_' + str(i))
        return G


class AutoRescopeRule(Rule):

    def __init__(self, L, VK):

        self.L = L
        nx.relabel_nodes(self.L, {n: '/'.join(['L-term', n]) for n in set(self.L.nodes) if n not in VK}, copy=False)

        nx.relabel_nodes(self.L, {n: '/'.join(['K-term', n]) for n in set(self.L.nodes) & VK}, copy=False)
        self.K = self.L.subgraph({'/'.join(['K-term', n]) for n in VK})

        self.L_K = self.L.subgraph(set(self.L.nodes).difference({'/'.join(['K-term', n]) for n in VK}))
        K_2_L = [e for e in set(self.L.edges).difference(set(self.L_K.edges) | set(self.K.edges)) if e[0] in self.K.nodes]
        L_2_K = [e for e in set(self.L.edges).difference(set(self.L_K.edges) | set(self.K.edges)) if e[1] in self.K.nodes]

        self.R_K = nx.relabel_nodes(copy.copy(self.L_K), {n: n.replace('L-term', 'R-term') for n in self.L_K.nodes})
        L_K_2_R_K_morph = get_morphisms(self.L_K, self.R_K)
        assert len(L_K_2_R_K_morph) == 1
        L_K_2_R_K = L_K_2_R_K_morph[0]
        self.K_2_R = [(u, L_K_2_R_K[v]) for u, v in K_2_L]
        self.R_2_K = [(L_K_2_R_K[u], v) for u, v in L_2_K]

        self.S = nx.compose(self.L, self.R_K)
        self.S.add_edges_from(self.K_2_R + self.R_2_K)

    @staticmethod
    def core(H_I):

        # automatically detect the scope of the operations involved (should be unique!)
        scopes = list(set(H_I.nodes[n]['scope'] for n in H_I.nodes if (H_I.nodes[n]['bipartite'] == __KERNEL_PARTITION__) and (H_I.nodes[n]['scope'] != '')))
        assert len(scopes) == 1
        new_scope = scopes[0]

        J_I = copy.copy(H_I)
        J_I = nx.relabel_nodes(J_I, {n: 'new_' + n for n in J_I.nodes})
        nx.set_node_attributes(J_I, {n: new_scope for n in J_I.nodes if (J_I.nodes[n]['bipartite'] == __KERNEL_PARTITION__)}, 'scope')
        return J_I

    def discover(self, G):
        return get_morphisms(G, self.L)

    def _apply(self, G, g):

        H_I = G.subgraph({k for k, v in g.items() if v not in self.K.nodes})
        J_I = self.core(H_I)

        J_I_2_R_K_morph = get_morphisms(J_I, self.R_K)
        assert len(J_I_2_R_K_morph) == 1
        J_I_2_R_K = J_I_2_R_K_morph[0]
        R_K_2_J_I = {v: k for k, v in J_I_2_R_K.items()}

        G = nx.compose(G, J_I)
        VI = {k for k, v in g.items() if v in self.K.nodes}
        for vI in VI:
            vK = g[vI]
            for e in [e for e in self.K_2_R if e[0] == vK]:
                G.add_edge(vI, R_K_2_J_I[e[1]])
            for e in [e for e in self.R_2_K if e[1] == vK]:
                G.add_edge(R_K_2_J_I[e[0]], vI)

        G.remove_nodes_from(H_I.nodes)

        return G

    def apply(self, G, gs):
        for g in gs:
            G = self._apply(G, g)
        return G


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
