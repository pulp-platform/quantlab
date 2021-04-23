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
        # 1. define the template graph L [L-term]
        # 2. define the context (sub-)graph K [K-term]
        # 3. define the template (sub-)graph L\K
        # 4. define the replacement (sub-)graph R\K
        # 5. define the connections between the context (sub-)graph K and the replacement (sub-)graph R\K to obtain the replacement graph R [R-term]
        # 6. build the `Seeker` object; since the L-term of the GRR is usually defined or at least modified in the body of a GRR's constructor method, it is safe to build the seeker at the end
        raise NotImplementedError

    def core(self, HI):
        # transform the match (sub-)graph H\I into the substitute (sub-)graph J\I
        raise NotImplementedError

    def apply(self, G, nodes_dict, g):
        # 1. create copies of G and nodes_dict
        # 2. characterise the match graph H: separate the interface (sub-)graph I from the match (sub-)graph H\I
        # 3. transform the match (sub-)graph H\I into the substitute (sub-)graph J\I
        # 4. glue the substitute (sub-)graph J\I to the main graph G via the interface (sub-)graph I
        # 5. discard the match (sub-)graph H\I (including all the arcs between its nodes and the nodes of the interface (sub-)graph I)
        # return G, nodes_dict  # should be copies!
        raise NotImplementedError

    def seek(self, G, nodes_dict):
        # return gs (a `list` of `dictionaries` whose keys are nodes of G and values are nodes of L
        raise NotImplementedError


class ManualRescopingRule(DPORule):

    def __init__(self, L, K, new_scope):

        # define the template graph L [L-term]
        VLK = set(L.nodes).difference(set(K.nodes))
        self.L = nx.relabel_nodes(L, {vLK: '/'.join(['L-term', vLK]) for vLK in VLK}, copy=True)

        # define the context (sub-)graph K [K-term]
        VK = set(self.L.nodes).intersection(set(K.nodes))
        nx.relabel_nodes(self.L, {vK: '/'.join(['K-term', vK]) for vK in VK}, copy=False)
        VK = {vL for vL in set(self.L.nodes) if vL.startswith('K-term')}
        self.K = self.L.subgraph(VK)

        # define the template (sub-)graph L\K
        VLK = set(self.L.nodes).difference(set(self.K.nodes))
        self.LK = self.L.subgraph(VLK)

        # get the arcs that go from the vertices of K to those of L\K, and viceversa
        E_K2LK2K = {arc for arc in set(self.L.edges).difference(set(self.LK.edges) | set(self.K.edges))}
        E_K2LK = {arc for arc in E_K2LK2K if arc[0] in set(self.K.nodes)}
        E_LK2K = {arc for arc in E_K2LK2K if arc[1] in set(self.K.nodes)}

        # define the replacement (sub-)graph R\K ["gluing" R\K to K yields the graph R, i.e., the R-term] -- rescoping rules implement one-to-one mappings
        self.RK = nx.relabel_nodes(self.LK, {vLK: vLK.replace('L-term', 'R-term') for vLK in set(self.LK.nodes)}, copy=True)

        # define the arcs that go from the vertices of K to those of R\K, and viceversa
        LK2RK_morphisms = Seeker(self.RK).get_morphisms(self.LK)
        assert len(LK2RK_morphisms) == 1
        g_LK2RK = LK2RK_morphisms[0]
        E_K2RK = {(u, g_LK2RK[v]) for u, v in E_K2LK}
        E_RK2K = {(g_LK2RK[u], v) for u, v in E_LK2K}
        E_K2RK2K = E_K2RK | E_RK2K
        # disintegrate `E_K2RK` and `E_RK2K` along fibres to speed up rule application
        self.F_K2RK = {vK: set(arc for arc in E_K2RK if arc[0] == vK) for vK in set(self.K.nodes)}
        self.F_RK2K = {vK: set(arc for arc in E_RK2K if arc[1] == vK) for vK in set(self.K.nodes)}

        # # glue together the (sub-)graphs L\K and R\K along the vertices of K
        # self.S = nx.compose(self.L, self.RK)
        # self.S.add_edges_from(E_K2RK2K)

        # since the GRR's L-term will not be modified from now on, I can safely build the seeker
        self.seeker = Seeker(self.L)

        # this machinery can generate always-new scope labels
        self._scope = new_scope
        self._counter = itertools.count()

    def _get_new_scope(self):
        new_scope = '.'.join([self._scope, '{:03d}'.format(next(self._counter))])
        return new_scope

    def core(self, HI):

        # generate a new scope
        new_scope = self._get_new_scope()

        # create a copy of the match (sub-)graph, but whose nodes have a new scope; its nodes are assigned different IDs to avoid conflicting IDs when gluing to G
        JI = nx.relabel_nodes(HI, {vHI: vHI.replace('__tmp__', '') for vHI in set(HI.nodes)}, copy=True)
        nx.set_node_attributes(JI, {vJI: new_scope for vJI in set(JI.nodes) if (JI.nodes[vJI]['bipartite'] == quantlib.graphs.graphs.__KERNEL_PARTITION__)}, 'scope')

        return JI

    def apply(self, G, nodes_dict, g):

        # create new containers
        G = G.copy()
        nodes_dict = {**nodes_dict}

        # mark the to-be-rescoped nodes' IDs as obsolete
        gkeys_2_tmpkeys = {vH: vH + '__tmp__' for vH, vL in g.items() if vL not in set(self.K.nodes)}
        nx.relabel_nodes(G, gkeys_2_tmpkeys, copy=False)
        # characterise the match graph H
        VI = {vH for vH, vL in g.items() if vL in set(self.K.nodes)}
        I = G.subgraph(VI)
        VHI = {gkeys_2_tmpkeys[vH] for vH, vL in g.items() if vL not in set(self.K.nodes)}
        HI = G.subgraph(VHI)

        # generate the substitute (sub-)graph J\I
        JI = self.core(HI)
        # add the substitute (sub-)graph J\I to the main graph G
        G = nx.compose(G, JI)

        # compute the morphism 'g_{(J \setminus I) \to (R \setminus K)}': I need it to glue J\I to I
        JI2RK_morphisms = Seeker(self.RK).get_morphisms(JI)
        assert len(JI2RK_morphisms) == 1
        g_JI2RK = JI2RK_morphisms[0]
        g_RK2JI = {vRK: vJI for vJI, vRK in g_JI2RK.items()}
        # glue the substitue (sub-)graph J\I to the main graph G
        for vI in set(I.nodes):
            vK = g[vI]
            G.add_edges_from({(vI, g_RK2JI[vRK]) for (_, vRK) in self.F_K2RK[vK]})
            G.add_edges_from({(g_RK2JI[vRK], vI) for (vRK, _) in self.F_RK2K[vK]})

        # discard the match (sub-)graph H\I; arcs between H\I and I are deleted automatically
        G.remove_nodes_from(set(HI.nodes))

        return G, nodes_dict

    def seek(self, G, nodes_dict):
        gs = self.seeker.get_morphisms(G)
        return gs


class AutoRescopingRule(DPORule):

    def __init__(self, L, K):

        # define the graph L [L-term]
        VL = set(L.nodes).difference(set(K.nodes))
        self.L = nx.relabel_nodes(L, {vL: '/'.join(['L-term', vL]) for vL in VL}, copy=True)

        # define the (sub-)graph K [K-term]
        VK = set(self.L.nodes).intersection(set(K.nodes))
        nx.relabel_nodes(self.L, {vK: '/'.join(['K-term', vK]) for vK in VK}, copy=False)
        VK = {vL for vL in set(self.L.nodes) if vL.startswith('K-term')}
        self.K = self.L.subgraph(VK)

        # define the (sub-)graph L\K
        VLK = set(self.L.nodes).difference(set(self.K.nodes))
        self.LK = self.L.subgraph(VLK)

        # get the arcs that go from the vertices of K to those of L\K, and viceversa
        E_K2LK2K = {arc for arc in set(self.L.edges).difference(set(self.LK.edges) | set(self.K.edges))}
        E_K2LK = {arc for arc in E_K2LK2K if arc[0] in set(self.K.nodes)}
        E_LK2K = {arc for arc in E_K2LK2K if arc[1] in set(self.K.nodes)}

        # define the (sub-)graph R\K ["gluing" R\K to K yields the graph R, i.e., the R-term] -- rescoping rules implement one-to-one mappings
        self.RK = nx.relabel_nodes(self.LK, {vLK: vLK.replace('L-term', 'R-term') for vLK in set(self.LK.nodes)}, copy=True)

        # define the arcs that go from the vertices of K to those of R\K, and viceversa
        LK2RK_morphisms = Seeker(self.RK).get_morphisms(self.LK)
        assert len(LK2RK_morphisms) == 1
        g_LK2RK = LK2RK_morphisms[0]
        E_K2RK = {(u, g_LK2RK[v]) for u, v in E_K2LK}
        E_RK2K = {(g_LK2RK[u], v) for u, v in E_LK2K}
        E_K2RK2K = E_K2RK | E_RK2K
        # disintegrate `E_K2RK` and `E_RK2K` along fibres to speed up rule application
        self.F_K2RK = {vK: set(arc for arc in E_K2RK if arc[0] == vK) for vK in set(self.K.nodes)}
        self.F_RK2K = {vK: set(arc for arc in E_RK2K if arc[1] == vK) for vK in set(self.K.nodes)}

        # # glue together the (sub-)graphs L\K and R\K along the vertices of K
        # self.S = nx.compose(self.L, self.RK)
        # self.S.add_edges_from(E_K2RK2K)

        # since the GRR's L-term will not be modified from now on, I can safely build the seeker
        self.seeker = Seeker(self.L)

    def core(self, HI):

        # automatically detect the scope of the operations involved (should be unique!)
        scopes = {HI.nodes[vHI]['scope'] for vHI in set(HI.nodes) if (HI.nodes[vHI]['bipartite'] == quantlib.graphs.graphs.__KERNEL_PARTITION__)}
        try:
            scopes.remove('')
        except KeyError:
            pass
        assert len(scopes) == 1  # up to now, quantlib's `nn.Module`s traces have included at least one correctly scoped operation... maybe we could suggest the user to apply a `ManualRescopingRule` when this does not happen?
        new_scope = list(scopes)[0]

        # create a copy of the match (sub-)graph, but whose nodes have a new scope; its nodes are assigned different IDs to avoid conflicting IDs when gluing to G
        JI = nx.relabel_nodes(HI, {vHI: vHI.replace('__tmp__', '') for vHI in set(HI.nodes)}, copy=True)
        nx.set_node_attributes(JI, {vJI: new_scope for vJI in JI.nodes if (JI.nodes[vJI]['bipartite'] == quantlib.graphs.graphs.__KERNEL_PARTITION__)}, 'scope')

        return JI

    def apply(self, G, nodes_dict, g):

        # create new containers
        G = G.copy()
        nodes_dict = {**nodes_dict}

        # mark the to-be-rescoped nodes' IDs as obsolete
        gkeys_2_tmpkeys = {vH: vH + '__tmp__' for vH, vL in g.items() if vL not in set(self.K.nodes)}
        nx.relabel_nodes(G, gkeys_2_tmpkeys, copy=False)
        # characterise the match graph H
        VI = {vH for vH, vL in g.items() if vL in set(self.K.nodes)}
        I = G.subgraph(VI)
        VHI = {gkeys_2_tmpkeys[vH] for vH, vL in g.items() if vL not in set(self.K.nodes)}
        HI = G.subgraph(VHI)

        # generate the substitute (sub-)graph J\I
        JI = self.core(HI)
        # add the substitute (sub-)graph J\I to the main graph G
        G = nx.compose(G, JI)

        # compute the morphism 'g_{(J \setminus I) \to (R \setminus K)}': I need it to glue J\I to I
        JI2RK_morphisms = Seeker(self.RK).get_morphisms(JI)
        assert len(JI2RK_morphisms) == 1
        g_JI2RK = JI2RK_morphisms[0]
        g_RK2JI = {vRK: vJI for vJI, vRK in g_JI2RK.items()}
        # glue the substitue (sub-)graph J\I to the main graph G
        for vI in set(I.nodes):
            vK = g[vI]
            G.add_edges_from({(vI, g_RK2JI[vRK]) for (_, vRK) in self.F_K2RK[vK]})
            G.add_edges_from({(g_RK2JI[vRK], vI) for (vRK, _) in self.F_RK2K[vK]})

        # discard the match (sub-)graph H\I; arcs between H\I and I are deleted automatically
        G.remove_nodes_from(set(HI.nodes))

        return G, nodes_dict

    def seek(self, G, nodes_dict):
        gs = self.seeker.get_morphisms(G)
        return gs
