import networkx as nx
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
        type = quantlib.graphs.graphs.HelperInput().__class__.__name__ if self._io == 'I' else quantlib.graphs.graphs.HelperOutput().__class__.__name__

        self.RK = nx.DiGraph()
        self.RK.add_nodes_from(set([''.join(['R-term/', 'H', self._io])]), bipartite=quantlib.graphs.graphs.__KERNEL_PARTITION__, type=type)

        self._counter = itertools.count()

    def core(self):

        vJI = ''.join(['H', self._io, '{:06d}'.format(next(self._counter))])
        JI = nx.relabel_nodes(self.RK, {vRK: vJI for vRK in set(self.RK.nodes)}, copy=True)

        m = quantlib.graphs.graphs.HelperInput() if self._io == 'I' else quantlib.graphs.graphs.HelperOutput()
        ptnode = quantlib.graphs.graphs.PyTorchNode(m)

        nx.set_node_attributes(JI, {vJI: '' for vJI in set(JI.nodes) if (JI.nodes[vJI]['bipartite'] == quantlib.graphs.graphs.__KERNEL_PARTITION__)}, 'scope')

        return JI, {vJI: ptnode}

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
        type = quantlib.graphs.graphs.HelperInput().__class__.__name__ if self._io == 'I' else quantlib.graphs.graphs.HelperOutput().__class__.__name__

        self.LK = nx.DiGraph()
        self.LK.add_nodes_from(set([''.join(['L-term/', 'H', self._io])]), bipartite=quantlib.graphs.graphs.__KERNEL_PARTITION__, type=type)

        self.seeker = Seeker(self.LK)

    def core(self):
        pass

    def apply(self, G, nodes_dict, g):

        G = G.copy()
        VHI = set(g.keys())
        HI = G.subgraph(VHI)
        G.remove_nodes_from(set(HI.nodes))

        nodes_dict = {**nodes_dict}  # copy the dictionary
        for n in set(HI.nodes):
            del nodes_dict[n]

        return G, nodes_dict

    def seek(self, G, nodes_dict):
        gs = self.seeker.get_morphisms(G)
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
        # a "precision tunnel" will be added after nodes of type 'type'

        self.KA = nx.DiGraph()
        self.KA.add_nodes_from(set([''.join(['K-term/', 'HPTin'])]), bipartite=quantlib.graphs.graphs.__KERNEL_PARTITION__, type=type)

        self.seeker = Seeker(self.KA)

    def core(self, H, VIin, eps):

        vH_2_vJI = {**{vH: vH.replace('O', 'HPTin') for vH in set(H.nodes).intersection(VIin)}, **{vH: vH.replace('O', 'HPTout') for vH in set(H.nodes).difference(VIin)}}
        JI = nx.relabel_nodes(H, vH_2_vJI, copy=True)
        nx.set_node_attributes(JI, {vJI: None for vJI in set(JI.nodes) if (JI.nodes[vJI]['bipartite'] == quantlib.graphs.graphs.__KERNEL_PARTITION__)}, 'type')

        vJI_2_m = {vJI: quantlib.graphs.graphs.HelperInputPrecisionTunnel(eps) if vJI.startswith('HPTin') else quantlib.graphs.graphs.HelperOutputPrecisionTunnel(eps) for vJI in set(JI.nodes)}  # TODO: what about data nodes?
        vJI_2_ptnode = {vJI: quantlib.graphs.graphs.PyTorchNode(m) for vJI, m in vJI_2_m.items()}

        nx.set_node_attributes(JI, {k: v.ntype for k, v in vJI_2_ptnode.items()}, 'type')
        nx.set_node_attributes(JI, {vJI: '' for vJI in set(JI.nodes) if (JI.nodes[vJI]['bipartite'] == quantlib.graphs.graphs.__KERNEL_PARTITION__)}, 'scope')

        return vH_2_vJI, JI, vJI_2_ptnode

    def apply(self, G, nodes_dict, g):

        G = G.copy()

        # find interface nodes
        VIin = set(g.keys())
        VIout = set(itertools.chain.from_iterable([set(G.successors(vI)) for vI in VIin]))
        VI = VIin | VIout

        # compute interface graph
        VH = VI
        H = G.subgraph(VH)
        # I = nx.subgraph_view(G, filter_node=show_nodes(VI), filter_edge=hide_edges(set(H.edges)))

        # create precision tunnel
        eps = nodes_dict[next(iter(VIin))].nobj.abs_max_value.item()  # TODO: only `STEActivation` nodes have `abs_max_value` attribute! try to homogenise this in the future
        vH_2_vJI, JI, vJI_2_ptnode = self.core(H, VIin, eps)

        # link J\I to I, then delete (H) \ (I)
        G = nx.compose(G, JI)
        E_I2JI = {(u, v) for u, v in vH_2_vJI.items() if u in VIin}
        E_JI2I = {(v, u) for u, v in vH_2_vJI.items() if u in VIout}
        E_I2JI2I = E_I2JI | E_JI2I
        G.add_edges_from(E_I2JI2I)
        G.remove_edges_from(set(H.edges))

        nodes_dict = {**nodes_dict, **vJI_2_ptnode}

        return G, nodes_dict

    def seek(self, G, nodes_dict):

        def is_valid_application_point(g):

            VIin = set(g.keys())
            VIout = set(itertools.chain.from_iterable([set(G.successors(vI)) for vI in VIin]))
            is_ok = len(VIout) > 0  # adding a precision tunnel is justified just in case the node has at least one output

            for vI in VIout:
                is_ok = set(G.predecessors(vI)).issubset(VIin)  # if an output node receives inputs also from another node, am I sure that I can remove the tunnel? better safe than sorry...

            return is_ok

        gs = self.seeker.get_morphisms(G)
        gs = list(filter(is_valid_application_point, gs))

        return gs


class RemovePrecisionTunnelRule(HelperRule):

    def __init__(self):

        self.LA = nx.DiGraph()
        self.LA.add_nodes_from(set([''.join(['L-term/', 'HPTin'])]), bipartite=quantlib.graphs.graphs.__KERNEL_PARTITION__, type=quantlib.graphs.graphs.HelperInputPrecisionTunnel(1.0).__class__.__name__)

        self.seeker = Seeker(self.LA)

    def apply(self, G, nodes_dict, g):

        G = G.copy()

        VHin = set(g.keys())
        VHout = set(G.successors(next(iter(VHin))))
        VH = VHin | VHout

        VIin = set(G.predecessors(next(iter(VHin))))
        VIout = set(itertools.chain.from_iterable([set(G.successors(vH)) for vH in VHout]))

        G.add_edges_from(list(itertools.product(VIin, VIout)))
        G.remove_nodes_from(VH)

        nodes_dict = {**nodes_dict}
        for n in VH:
            del nodes_dict[n]

        return G, nodes_dict

    def seek(self, G, nodes_dict):

        def is_valid_application_point(g):

            VHin = set(g.keys())
            assert len(VHin) == 1
            VHout = set(G.successors(next(iter(VHin))))

            # this GRR is meant to be the "inverse" of the 'AddPrecisionTunnel' GRR
            # no other GRRs except for these two should be allowed to add/remove `HelperPrecisionTunnel` nodes
            assert all([G.nodes[vH]['type'] == quantlib.graphs.graphs.HelperOutputPrecisionTunnel(1.0).__class__.__name__ for vH in VHout])

            epss_in = {nodes_dict[next(iter(VHin))].nobj.eps_in}
            epss_out = {nodes_dict[vH].nobj.eps_out for vH in VHout}

            print(epss_in, epss_out, set(g.keys()))

            is_ok = len(epss_in | epss_out) == 1  # all the output quanta must agree with the input quantum

            return is_ok

        gs = self.seeker.get_morphisms(G)
        gs = list(filter(is_valid_application_point, gs))

        return gs
