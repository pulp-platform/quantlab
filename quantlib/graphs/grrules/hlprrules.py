import networkx as nx
from networkx.classes.filters import show_nodes, hide_edges
import itertools

from .seeker import Seeker
import quantlib.graphs.graphs


__all__ = [
    'AddInputNodeRule',
    'AddOutputNodeRule',
    'RemoveInputNodeRule',
    'RemoveOutputNodeRule',
    'AddPrecisionTunnelRule',
    'RemovePrecisionTunnelRule',
]


class HelperRule(object):
    """A GRR that prepares the graph for the 'core' editing.

    This GRR inserts nodes into the computational graph that are propedeutics
    to the application of other GRRs; or, after all the 'core' GRRs have been
    applied, it can remove specific subgraphs that could be replaced by an
    identity operation or by groups of identity operations, and are therefore
    redundant for computational purposes.

    This GRR still follows the algebraic approach to graph rewriting, but the
    application points are computed 'on-the-fly'. In this sense, the rule
    actually implements a vast (possibly infinite) set of GRRs.
    """
    def __init__(self):
        raise NotImplementedError

    def core(self):
        raise NotImplementedError

    def apply(self, G, nodes_dict, g):
        # return G, nodes_dict
        raise NotImplementedError

    def seek(self, G, nodes_dict):
        # return gs
        raise NotImplementedError


######################
## I/O HELPER NODES ##
######################

class AddIONodeRule(HelperRule):

    def __init__(self, io):

        self._io = io  # either 'I' or 'O'
        type = quantlib.graphs.graphs.HelperInput.__name__ if self._io == 'I' else quantlib.graphs.graphs.HelperOutput.__name__

        self.RK = nx.DiGraph()
        self.RK.add_nodes_from(set([''.join(['R-term/', 'H', self._io])]), bipartite=quantlib.graphs.graphs.__KERNEL_PARTITION__, type=type)

        self._counter = itertools.count()

    def core(self):

        vJI = ''.join(['H', self._io, quantlib.graphs.graphs.__NODE_ID_FORMAT__.format(next(self._counter))])
        JI = nx.relabel_nodes(self.RK, {vRK: vJI for vRK in set(self.RK.nodes)}, copy=True)

        m = quantlib.graphs.graphs.HelperInput() if self._io == 'I' else quantlib.graphs.graphs.HelperOutput()
        vJI_2_ptnode = {vJI: quantlib.graphs.graphs.PyTorchNode(m)}

        nx.set_node_attributes(JI, {vJI: '' for vJI in set(JI.nodes) if (JI.nodes[vJI]['bipartite'] == quantlib.graphs.graphs.__KERNEL_PARTITION__)}, 'scope')

        return JI, vJI_2_ptnode

    def apply(self, G, nodes_dict, g):

        G = G.copy()

        VI = set(g.keys())
        I = G.subgraph(VI)

        JI, vJI_2_ptnode = self.core()

        G = nx.compose(G, JI)
        if self._io == 'I':
            E_JI2I = set(itertools.product(set(JI.nodes), set(I.nodes)))
            G.add_edges_from(E_JI2I)
        elif self._io == 'O':
            E_I2JI = set(itertools.product(set(I.nodes), set(JI.nodes)))
            G.add_edges_from(E_I2JI)

        nodes_dict = {**nodes_dict, **vJI_2_ptnode}

        return G, nodes_dict

    def seek(self, G, nodes_dict, VIs):

        if self._io == 'I':
            VIs = list(filter(lambda VI: len(set(itertools.chain.from_iterable([set(G.predecessors(vI)) for vI in VI]))) == 0, VIs))
        if self._io == 'O':
            assert len(set(itertools.chain.from_iterable(VIs))) == sum([len(VI) for VI in VIs])  # I assume that an operation can't write to multiple output nodes
            VIs = list(filter(lambda VI: len(set(itertools.chain.from_iterable([set(G.successors(vI)) for vI in VI]))) == 0, VIs))

        gs = []
        for VI in VIs:
            g = {vI: None for vI in VI}  # there is no fixed context term (K-term) for this rule!
            gs.append(g)

        return gs


class AddInputNodeRule(AddIONodeRule):

    def __init__(self):
        super(AddInputNodeRule, self).__init__('I')


class AddOutputNodeRule(AddIONodeRule):

    def __init__(self):
        super(AddOutputNodeRule, self).__init__('O')


class RemoveIONodeRule(HelperRule):

    def __init__(self, io):

        self._io = io  # either 'I' or 'O'
        type = quantlib.graphs.graphs.HelperInput.__name__ if self._io == 'I' else quantlib.graphs.graphs.HelperOutput.__name__

        # the I/O operation will serve as an "anchor"; from it, I will be able to (implicitly) generate and apply the graph rewriting rule on-the-fly
        self.LK = nx.DiGraph()
        self.LK.add_nodes_from(set([''.join(['L-term/', 'H', self._io])]), bipartite=quantlib.graphs.graphs.__KERNEL_PARTITION__, type=type)

        self.seeker = Seeker(self.LK)

    def core(self):
        pass

    def apply(self, G, nodes_dict, g):

        # create new containers
        G = G.copy()
        nodes_dict = {**nodes_dict}

        # characterise the match (sub-)graph H\I
        VHI = set(g.keys())
        HI = G.subgraph(VHI)

        # delete the I/O nodes
        G.remove_nodes_from(set(HI.nodes))
        for n in set(HI.nodes):
            del nodes_dict[n]

        return G, nodes_dict

    def seek(self, G, nodes_dict):

        def is_valid_application_point(g):

            VJI = set(g.keys())
            assert len(VJI) == 1

            if self._io == 'I':
                is_ok = len(set(G.predecessors(next(iter(VJI))))) == 0  # an input node does not read the output of any other node
            elif self._io == 'O':
                is_ok = len(set(G.successors(next(iter(VJI))))) == 0  # an output node's output won't be read by any node

            return is_ok

        gs = self.seeker.get_morphisms(G)
        gs = list(filter(is_valid_application_point, gs))

        return gs


class RemoveInputNodeRule(RemoveIONodeRule):

    def __init__(self):
        super(RemoveInputNodeRule, self).__init__('I')


class RemoveOutputNodeRule(RemoveIONodeRule):

    def __init__(self):
        super(RemoveOutputNodeRule, self).__init__('O')


############################
## PRECISION HELPER NODES ##
############################

class AddPrecisionTunnelRule(HelperRule):

    def __init__(self, type):
        """Insert a 'precision tunnel' after idempotent operations.

        Imagine having a function composition :math:`o \circ f \circ i`,
        which needs to be transformed into a second composition
        :math:`h \circ g`. Suppose that the information required to compute
         :math:`h` is contained in :math:`o` and (partly) in :math:`f`, i.e.,
        there exists a transformation :math:`T_{h} \,|\, h = T_{h}(o, f)`;
        similarly, the information required to compute :math:`g` is contained
        (partly) in :math:`f` and in :math:`i`, i.e., there exists a transform
        :math:`T_{g} \,|\, g = T_{g}(f, i)`. We assume that :math:`T_{h}` and
        :math:`T_{g}` can be applied in any order, but must be executed
        sequentially; also, we assume that after the application of a
        transform its inputs will be destroyed. We see then that there is no
        valid order, since each of them will destroy :math:`f`, preventing the
        application of the second transformation.

        If we suppose that :math:`f` is idempotent (i.e., it is such that
        :math:`f \circ f = f`), we can rewrite the original term as
        :math:`o \circ f \circ f \circ i` before applying the transformation
        rules :math:`T_{h}` and :math:`T_{g}`. In this case, we can derive the
        desired form :math:`T_{h}(o, f) \circ T_{g}(f, i)` without any issue.

        In particular, we focus on the case where :math:`f` is a quantization
        operator, i.e., an activation function of the form
        :math:`f = e \circ r_{p} \circ e^{-1}`, where :math:`r_{p}` is a
        rounding operation at precision :math:`p`, and :math:`e` is an
        element-wise multiplication by a positive number (possibly represented
        in floating point). We assume that the 'anchor' of each application
        point (i.e., the quantization operator :math:`f`) will receive data
        from just one operation :math:`i`, but its outputs will be read by a
        positive number of operations :math:`o^{(k)}, k = 0, \dots, K-1`, for
        some positive integer :math:`K`. Then, we can rewrite each sequence
        :math:`o^{(k)} \circ f \circ i` as
        :math:`o^{(k)} \circ e \circ r_{p} \circ e^{-1} \circ e \circ r_{p} \circ e^{-1} \circ i`.
        In this way, we will be able to apply :math:`T_{h}` :math:`K` times,
        returning :math:`h^{(k)} = T_{h}(o^{(k)}, f)`, and :math:`T_{g}` just
        once, returning :math:`g = T_{g}(f, i)`.
        """

        # the idempotent operation will serve as an "anchor"; from it, I will be able to (implicitly) generate and apply the graph rewriting rule on-the-fly
        self.Kin = nx.DiGraph()
        self.Kin.add_nodes_from(set([''.join(['K-term/', 'HPTin'])]), bipartite=quantlib.graphs.graphs.__KERNEL_PARTITION__, type=type)

        self.seeker = Seeker(self.Kin)

    def core(self, H, Iin, Iout, eps, nodes_dict):

        # `HelperPrecisionTunnel` nodes are meant to serve as 'stubs' when applying full-fledged GRRs;
        # this graph will ensure the correct connectivity between the 'pieces' of the full graph that will be derived by applying full-fledged GRRs
        vH_2_vJI_PTin = {vH: vH.replace('O', 'HPTin') for vH in set(H.nodes).intersection(set(Iin.nodes))}
        vH_2_vJI_PTout = {vH: vH.replace('O', 'HPTout') for vH in set(H.nodes).intersection(set(Iout.nodes))}
        JI = nx.relabel_nodes(H, {**vH_2_vJI_PTin, **vH_2_vJI_PTout}, copy=True)
        nx.set_node_attributes(JI, {vJI: quantlib.graphs.graphs.HelperInputPrecisionTunnel.__name__ for vJI in set(vH_2_vJI_PTin.values())}, 'type')
        nx.set_node_attributes(JI, {vJI: quantlib.graphs.graphs.HelperOutputPrecisionTunnel.__name__ for vJI in set(vH_2_vJI_PTout.values())}, 'type')

        # replicate the "anchor" idempotent operation along each connection
        vJI_PTout_2_vJI_PTclone = {vJI: vJI.replace('HPTout', 'HPTclone') for vJI in set(vH_2_vJI_PTout.values())}
        for u, v in vJI_PTout_2_vJI_PTclone.items():
            Iin_clone = nx.relabel_nodes(H.subgraph(Iin), {next(iter(set(Iin.nodes))): v}, copy=True)
            JI = nx.compose(JI, Iin_clone)
            JI.add_edge(u, v)

        nx.set_node_attributes(JI, {vJI: '' for vJI in set(JI.nodes) if (JI.nodes[vJI]['bipartite'] == quantlib.graphs.graphs.__KERNEL_PARTITION__)}, 'scope')

        # compute the connections of the new nodes to the old nodes
        E_I2JI = {(vI, vJI) for vI, vJI in vH_2_vJI_PTin.items()}
        vJI_PTclone_2_vJI_PTout = {v: k for k, v in vJI_PTout_2_vJI_PTclone.items()}
        vJI_PTout_2_vH = {v: k for k, v in vH_2_vJI_PTout.items()}
        E_JI2I = {(vJI, vJI_PTout_2_vH[vJI_PTclone_2_vJI_PTout[vJI]]) for vJI in set(vJI_PTclone_2_vJI_PTout.keys())}

        # register the technical specs of the new ops
        vJI_2_ptnode = {}
        for vJI in set(JI.nodes):
            if JI.nodes[vJI]['type'] == quantlib.graphs.graphs.HelperInputPrecisionTunnel.__name__:
                ptnode = quantlib.graphs.graphs.PyTorchNode(quantlib.graphs.graphs.HelperInputPrecisionTunnel(eps))
            elif JI.nodes[vJI]['type'] == quantlib.graphs.graphs.HelperOutputPrecisionTunnel.__name__:
                ptnode = quantlib.graphs.graphs.PyTorchNode(quantlib.graphs.graphs.HelperOutputPrecisionTunnel(eps))
            else:
                ptnode = nodes_dict[next(iter(set(Iin.nodes)))]  # since the idempotent operation already exists, I just need a pointer to it
            vJI_2_ptnode[vJI] = ptnode

        return JI, vJI_2_ptnode, E_I2JI, E_JI2I

    def apply(self, G, nodes_dict, g):

        # create new containers
        G = G.copy()
        nodes_dict = {**nodes_dict}

        # compute the match graph H on-the-fly
        VIin = set(g.keys())
        VIout = set(G.successors(next(iter(VIin))))
        VI = VIin | VIout
        VH = VI
        H = G.subgraph(VH)
        I = nx.subgraph_view(G, filter_node=show_nodes(VI), filter_edge=hide_edges(set(H.edges)))
        Iin = I.subgraph(VIin)
        Iout = I.subgraph(VIout)

        # create the precision tunnel
        n = nodes_dict[next(iter(VIin))].nobj.num_levels
        m = nodes_dict[next(iter(VIin))].nobj.abs_max_value.item()  # TODO: only `STEActivation` nodes have `abs_max_value` attribute! try to homogenise this in the future
        eps = (2 * m) / (n - 1)
        JI, vJI_2_ptnode, E_I2JI, E_JI2I = self.core(H, Iin, Iout, eps, nodes_dict)

        # link the substitute (sub-)graph J\I to the interface (sub-)graph I
        G = nx.compose(G, JI)
        E_I2JI2I = E_I2JI | E_JI2I
        G.add_edges_from(E_I2JI2I)
        nodes_dict.update(vJI_2_ptnode)

        # delete H \ I (this it is NOT the match (sub-)graph H\I, but the difference between the match graph H and the interface (sub-)graph I)
        G.remove_edges_from(set(H.edges))

        return G, nodes_dict

    def seek(self, G, nodes_dict):

        def is_valid_application_point(g):

            VIin = set(g.keys())
            assert len(VIin) == 1
            VIout = set(G.successors(next(iter(VIin))))

            is_ok = len(VIout) > 0  # adding a precision tunnel makes sense just in case the node has at least one output
            return is_ok

        gs = self.seeker.get_morphisms(G)
        gs = list(filter(is_valid_application_point, gs))

        return gs


class RemovePrecisionTunnelRule(HelperRule):

    def __init__(self):
        """Delete the `HelperPrecisionTunnel` nodes in the graph.

        This GRR is not mean to act as a full inverse of the
        `AddPrecisionTunnelRule` GRR. It will only remove nodes whose
        corresponding `nn.Module`s are of type `HelperPrecisionTunnel`; it
        will not take care of 'reabsorbing' the copies of the idempotent
        operations generated by applications of the `AddPrecisionTunnelRule`.
        In fact, this GRR assumes that all those copies will be consumed by
        full-fledged GRRs. Since `HelperPrecisionTunnel` modules are meant to
        serve as 'stubs' when applying full-fledged GRRs, this rule should be
        applied only after all such copies will have been absorbed. In
        summary, this GRR is meant to 'clean up' the computational graph once
        all the 'core' GRRs will have been applied.
        """

        # the input to the precision tunnel will serve as an "anchor"; once I locate such a node, I will be able to (implicitly) generate and apply the graph rewriting rule on-the-fly
        self.LK = nx.DiGraph()
        self.LK.add_nodes_from(set([''.join(['L-term/', 'HPTin'])]), bipartite=quantlib.graphs.graphs.__KERNEL_PARTITION__, type=quantlib.graphs.graphs.HelperInputPrecisionTunnel(1.0).__class__.__name__)

        self.seeker = Seeker(self.LK)

    def apply(self, G, nodes_dict, g):

        # create new containers
        G = G.copy()
        nodes_dict = {**nodes_dict}

        # characterise the match graph H
        VHIin = set(g.keys())
        VHIout = set(G.successors(next(iter(VHIin))))
        VHI = VHIin | VHIout
        VIin = set(G.predecessors(next(iter(VHIin))))
        VIout = set(itertools.chain.from_iterable([set(G.successors(vH)) for vH in VHIout]))

        # add J \ I; (this it is NOT the substitute (sub-)graph J\I, but the difference between the substitute graph J and the interface (sub-)graph I)
        G.add_edges_from(list(itertools.product(VIin, VIout)))

        # delete the 'precision tunnel' nodes
        G.remove_nodes_from(VHI)
        for n in VHI:
            del nodes_dict[n]

        return G, nodes_dict

    def seek(self, G, nodes_dict):

        def is_valid_application_point(g):

            VHin = set(g.keys())
            assert len(VHin) == 1
            VHout = set(G.successors(next(iter(VHin))))
            assert all([G.nodes[vH]['type'] == quantlib.graphs.graphs.HelperOutputPrecisionTunnel.__name__ for vH in VHout])

            epss_in = {nodes_dict[next(iter(VHin))].nobj.eps_in}
            epss_out = {nodes_dict[vH].nobj.eps_out for vH in VHout}
            is_ok = len(epss_in | epss_out) == 1  # all the output quanta must agree with the input quantum

            return is_ok

        gs = self.seeker.get_morphisms(G)
        gs = list(filter(is_valid_application_point, gs))

        return gs
