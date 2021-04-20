import networkx as nx
import itertools

from .seeker import Seeker
import quantlib.graphs.graphs


__all__ = [
    'ManualRescopingRule',
    'AutoRescopingRule',
]


class DPORule(object):
    """This object represents an abstract *graph rewriting rule* (GRR).

    This process is similar to biological reactions between enzymes and other
    proteins or ensembles of proteins (the so-called "substrates"). Indeed,
    when an enzyme (the `Rule` object) encounters a substrate (an `nx.DiGraph`
    object) it looks for suitable binding sites (the application points of the
    rule) where it can catalyse a chemical reaction (an application of the
    rule yielding a transformed `nx.DiGraph`).
    """
    def __init__(self):
        # self.seeker = None  # since the L-term of the GRR is usually defined or at least modified in the GRR constructor method, the seeker should be built at its end
        raise NotImplementedError

    def core(self, HI):
        # transform H\I into J\I
        raise NotImplementedError

    def apply(self, G, nodes_dict, g):
        # identify interface I and non-interface H\I; transform H\I into J\I; glue J\I to I, then discard H\I
        # return G, nodes_dict
        raise NotImplementedError

    def seek(self, G, nodes_dict):
        # return gs
        raise NotImplementedError


class ManualRescopingRule(DPORule):

    def __init__(self, L, K, new_scope):

        # define L-term
        VLK = set(L.nodes).difference(set(K.nodes))
        self.L = nx.relabel_nodes(L, {vLK: '/'.join(['L-term', vLK]) for vLK in VLK}, copy=True)

        # define K-term
        VK = set(self.L.nodes).intersection(set(K.nodes))
        nx.relabel_nodes(self.L, {vK: '/'.join(['K-term', vK]) for vK in VK}, copy=False)
        VK = {vL for vL in set(self.L.nodes) if vL.startswith('K-term')}
        self.K = self.L.subgraph(VK)

        # get the graph L\K
        VLK = set(self.L.nodes).difference(set(self.K.nodes))
        self.LK = self.L.subgraph(VLK)
        # get the arcs between the vertices of K and the vertices of L\K and viceversa
        E_K2LK2K = {arc for arc in set(self.L.edges).difference(set(self.LK.edges) | set(self.K.edges))}
        E_K2LK = {arc for arc in E_K2LK2K if arc[0] in set(self.K.nodes)}
        E_LK2K = {arc for arc in E_K2LK2K if arc[1] in set(self.K.nodes)}

        # define the graph R\K -- rescoping rules implement one-to-one mappings
        self.RK = nx.relabel_nodes(self.LK, {vLK: vLK.replace('L-term', 'R-term') for vLK in set(self.LK.nodes)}, copy=True)
        # define the edges between the vertices of K and the vertices of R\K
        LK2RK_morphisms = Seeker(self.RK).get_morphisms(self.LK)
        assert len(LK2RK_morphisms) == 1
        g_LK2RK = LK2RK_morphisms[0]
        E_K2RK = {(u, g_LK2RK[v]) for u, v in E_K2LK}
        E_RK2K = {(g_LK2RK[u], v) for u, v in E_LK2K}
        E_K2RK2K = E_K2RK | E_RK2K
        # disintegrate `E_K2RK` and `E_RK2K` along fibres to speed up rule application
        self.F_K2RK = {vK: set(arc for arc in E_K2RK if arc[0] == vK) for vK in set(self.K.nodes)}
        self.F_RK2K = {vK: set(arc for arc in E_RK2K if arc[1] == vK) for vK in set(self.K.nodes)}

        # glue together the graphs L\K and R\K along the vertices in K
        self.S = nx.compose(self.L, self.RK)
        self.S.add_edges_from(E_K2RK2K)

        # since the GRR's L-term has been modified, rebuild the seeker
        self.seeker = Seeker(self.L)

        # generating machinery for scope labels
        self._new_scope = new_scope
        self._counter = itertools.count()

    def _get_new_scope(self):
        new_scope = '.'.join([self._new_scope, '{:03d}'.format(next(self._counter))])
        return new_scope

    def core(self, HI):

        new_scope = self._get_new_scope()

        JI = nx.relabel_nodes(HI, {vHI: vHI.replace('__tmp__', '') for vHI in set(HI.nodes)}, copy=True)
        nx.set_node_attributes(JI, {vJI: new_scope for vJI in set(JI.nodes) if (JI.nodes[vJI]['bipartite'] == quantlib.graphs.graphs.__KERNEL_PARTITION__)}, 'scope')

        return JI

    def apply(self, G, nodes_dict, g):

        gkeys_2_tmpkeys = {vH: vH + '__tmp__' for vH, vL in g.items() if vL not in set(self.K.nodes)}
        G = nx.relabel_nodes(G, gkeys_2_tmpkeys, copy=True)

        # characterise match graph H
        VI = {vH for vH, vL in g.items() if vL in set(self.K.nodes)}
        I = G.subgraph(VI)
        VHI = {gkeys_2_tmpkeys[vH] for vH, vL in g.items() if vL not in set(self.K.nodes)}
        HI = G.subgraph(VHI)

        # generate non-interface replacement graph JI
        JI = self.core(HI)

        # get morphism 'g_{(J \setminus I) \to (R \setminus K)}'
        JI2RK_morphisms = Seeker(self.RK).get_morphisms(JI)
        assert len(JI2RK_morphisms) == 1
        g_JI2RK = JI2RK_morphisms[0]
        g_RK2JI = {vRK: vJI for vJI, vRK in g_JI2RK.items()}

        # derive new graph
        G = nx.compose(G, JI)  # add non-interface replacement graph
        for vI in set(I.nodes):  # glue replacement graph to old graph
            vK = g[vI]
            G.add_edges_from({(vI, g_RK2JI[vRK]) for (_, vRK) in self.F_K2RK[vK]})
            G.add_edges_from({(g_RK2JI[vRK], vI) for (vRK, _) in self.F_RK2K[vK]})
        G.remove_nodes_from(set(HI.nodes))  # discard non-interface nodes of match graph

        nodes_dict = {**nodes_dict}

        return G, nodes_dict

    def seek(self, G, nodes_dict):
        gs = self.seeker.get_morphisms(G)
        return gs


class AutoRescopingRule(DPORule):

    def __init__(self, L, K):

        # define L-term
        VL = set(L.nodes).difference(set(K.nodes))
        self.L = nx.relabel_nodes(L, {vL: '/'.join(['L-term', vL]) for vL in VL}, copy=True)

        # define K-term
        VK = set(self.L.nodes).intersection(set(K.nodes))
        nx.relabel_nodes(self.L, {vK: '/'.join(['K-term', vK]) for vK in VK}, copy=False)
        VK = {vL for vL in set(self.L.nodes) if vL.startswith('K-term')}
        self.K = self.L.subgraph(VK)

        # get the graph L\K
        VLK = set(self.L.nodes).difference(set(self.K.nodes))
        self.LK = self.L.subgraph(VLK)
        # get the arcs that go from the vertices of K to those of L\K, and viceversa
        E_K2LK2K = {arc for arc in set(self.L.edges).difference(set(self.LK.edges) | set(self.K.edges))}
        E_K2LK = {arc for arc in E_K2LK2K if arc[0] in set(self.K.nodes)}
        E_LK2K = {arc for arc in E_K2LK2K if arc[1] in set(self.K.nodes)}

        # define the graph R\K -- rescoping rules implement one-to-one mappings
        self.RK = nx.relabel_nodes(self.LK, {vLK: vLK.replace('L-term', 'R-term') for vLK in set(self.LK.nodes)}, copy=True)
        # define the edges between the vertices of K and the vertices of R\K
        LK2RK_morphisms = Seeker(self.RK).get_morphisms(self.LK)
        assert len(LK2RK_morphisms) == 1
        g_LK2RK = LK2RK_morphisms[0]
        E_K2RK = {(u, g_LK2RK[v]) for u, v in E_K2LK}
        E_RK2K = {(g_LK2RK[u], v) for u, v in E_LK2K}
        E_K2RK2K = E_K2RK | E_RK2K
        # disintegrate `E_K2RK` and `E_RK2K` along fibres to speed up rule application
        self.F_K2RK = {vK: set(arc for arc in E_K2RK if arc[0] == vK) for vK in set(self.K.nodes)}
        self.F_RK2K = {vK: set(arc for arc in E_RK2K if arc[1] == vK) for vK in set(self.K.nodes)}

        # glue together the graphs L\K and R\K along the vertices in K
        self.S = nx.compose(self.L, self.RK)
        self.S.add_edges_from(E_K2RK2K)

        # since the GRR's L-term has been modified, rebuild the seeker
        self.seeker = Seeker(self.L)

    def core(self, HI):

        # automatically detect the scope of the operations involved (should be unique!)
        scopes = {HI.nodes[vHI]['scope'] for vHI in set(HI.nodes) if (HI.nodes[vHI]['bipartite'] == quantlib.graphs.graphs.__KERNEL_PARTITION__)}
        try:
            scopes.remove('')
        except KeyError:
            pass
        assert len(scopes) == 1  # up to now, quantlib's `nn.Module`s traces have included at least one correctly scoped operation...
        new_scope = list(scopes)[0]

        JI = nx.relabel_nodes(HI, {vHI: vHI.replace('__tmp__', '') for vHI in set(HI.nodes)}, copy=True)
        nx.set_node_attributes(JI, {vJI: new_scope for vJI in JI.nodes if (JI.nodes[vJI]['bipartite'] == quantlib.graphs.graphs.__KERNEL_PARTITION__)}, 'scope')

        return JI

    def apply(self, G, nodes_dict, g):

        gkeys_2_tmpkeys = {vH: vH + '__tmp__' for vH, vL in g.items() if vL not in set(self.K.nodes)}
        G = nx.relabel_nodes(G, gkeys_2_tmpkeys, copy=True)

        # characterise match graph H
        VI = {vH for vH, vL in g.items() if vL in set(self.K.nodes)}
        I = G.subgraph(VI)
        VHI = {gkeys_2_tmpkeys[vH] for vH, vL in g.items() if vL not in set(self.K.nodes)}
        HI = G.subgraph(VHI)

        # generate non-interface replacement graph JI
        JI = self.core(HI)

        # get morphism 'g_{(J \setminus I) \to (R \setminus K)}'
        JI2RK_morphisms = Seeker(self.RK).get_morphisms(JI)
        assert len(JI2RK_morphisms) == 1
        g_JI2RK = JI2RK_morphisms[0]
        g_RK2JI = {vRK: vJI for vJI, vRK in g_JI2RK.items()}

        # derive new graph
        G = nx.compose(G, JI)  # add non-interface replacement graph
        for vI in I.nodes:  # glue replacement graph to old graph
            vK = g[vI]
            G.add_edges_from({(vI, g_RK2JI[vRK]) for (_, vRK) in self.F_K2RK[vK]})
            G.add_edges_from({(g_RK2JI[vRK], vI) for (vRK, _) in self.F_RK2K[vK]})
        G.remove_nodes_from(set(HI.nodes))  # detach non-interface match graph

        nodes_dict = {**nodes_dict}

        return G, nodes_dict

    def seek(self, G, nodes_dict):
        gs = self.seeker.get_morphisms(G)
        return gs
