import networkx as nx
from networkx.algorithms import isomorphism
import copy

from quantlib.graphs.morph import __KERNEL_PARTITION__


__all__ = [
    'ManualRescopeRule',
    'AutoRescopeRule',
]


class Seeker(object):
    """This object looks for the application points of a graph rewriting rule.
    """
    def __init__(self, T):
        self.T = T

    @staticmethod
    def is_morphism(T, G, g):

        is_ok = False

        for vH, vT in g.items():

            is_same_partition = G.nodes[vH]['bipartite'] == T.nodes[vT]['bipartite']
            is_same_type = G.nodes[vH]['type'] == T.nodes[vT]['type']
            is_ok = is_same_partition and is_same_type  # computational graphs are node-labelled graphs, where node types act as labels

            if not is_ok:
                break

        return is_ok

    def get_morphisms(self, G):

        # In principle, morphisms do not need to be isomorphisms: this is a
        # restriction that I chose to simplify the work on QNNs conversion (it
        # makes solving ambiguities much easier).
        # In particular, candidate matchings will be induced subgraph
        # isomorphisms, not "spurious" monomorphisms:
        #
        #     https://github.com/networkx/networkx/blob/master/networkx/algorithms/isomorphism/isomorphvf2.py .
        #
        matcher = isomorphism.DiGraphMatcher(G, self.T)
        isomorphisms = list(matcher.subgraph_isomorphisms_iter())

        # check the second morphism condition (label consistency)
        morphisms = [g for g in isomorphisms if Seeker.is_morphism(self.T, G, g)]

        # remove duplicate morphisms
        unique_VHs = set(frozenset(g.keys()) for g in morphisms)
        VHs_2_morphisms = {VH: [g for g in morphisms if frozenset(g.keys()) == VH] for VH in unique_VHs}
        morphisms = [v[0] for v in VHs_2_morphisms.values()]

        return morphisms


class Rule(object):
    """This object represents an abstract *graph rewriting rule* (GRR).

    This process is similar to biological reactions between enzymes and other
    proteins or ensembles of proteins (the so-called "substrates"). Indeed,
    when an enzyme (the `Rule` object) encounters a substrate (an `nx.DiGraph`
    object) it looks for suitable binding sites (the application points of the
    rule) where it can catalyse a chemical reaction (an application of the
    rule yielding a transformed `nx.DiGraph`).
    """
    def __init__(self, L, K):
        self.seeker = Seeker(L)  # if the L-term of the GRR is modified in the GRR constructor method, the seeker should be rebuilt too
        pass

    @staticmethod
    def core(HI):
        # JI = core_transform(HI)
        # return JI
        pass

    def seek(self, G):
        return self.seeker.get_morphisms(G)

    def apply(self, G, g):
        # G = modify(G, g)
        return G

    def multiapply(self, G, gs):
        # for g in gs:
        #     self.apply(G, g)
        return G


class ManualRescopeRule(Rule):

    def __init__(self, L, K):
        super(ManualRescopeRule, self).__init__(L, K)

        # define L-term
        self.L = L
        nx.relabel_nodes(self.L, {n: '/'.join(['L-term', n]) for n in set(self.L.nodes).difference(set(K.nodes))}, copy=False)

        # define K-term
        nx.relabel_nodes(self.L, {n: '/'.join(['K-term', n]) for n in set(self.L.nodes).intersection(set(K.nodes))}, copy=False)
        self.K = self.L.subgraph({n for n in set(self.L.nodes) if n.startswith('K-term')})

        # get the graph L\K
        self.LK = self.L.subgraph(set(self.L.nodes).difference({n for n in set(self.L.nodes) if n.startswith('K-term')}))
        # get the edges between the vertices of K and the vertices of L\K
        E_K2LK2K = set(e for e in set(self.L.edges).difference(set(self.LK.edges) | set(self.K.edges)))
        E_K2LK = set(e for e in E_K2LK2K if e[0] in self.K.nodes)
        E_LK2K = set(e for e in E_K2LK2K if e[1] in self.K.nodes)

        # define the graph R\K -- rescoping rules implement one-to-one mappings
        self.RK = nx.relabel_nodes(copy.copy(self.LK), {n: n.replace('L-term', 'R-term') for n in set(self.LK.nodes)})
        # define the edges between the vertices of K and the vertices of R\K
        LK2RK_morphisms = Seeker(self.RK).get_morphisms(self.LK)
        assert len(LK2RK_morphisms) == 1
        g_LK2RK = LK2RK_morphisms[0]
        E_K2RK = set((u, g_LK2RK[v]) for u, v in E_K2LK)
        E_RK2K = set((g_LK2RK[u], v) for u, v in E_LK2K)
        E_K2RK2K = E_K2RK | E_RK2K
        # disintegrate `E_K2RK` and `E_RK2K` along fibres to speed up rule application
        self.F_K2RK = {vK: set(e for e in E_K2RK if e[0] == vK) for vK in self.K.nodes}
        self.F_RK2K = {vK: set(e for e in E_RK2K if e[1] == vK) for vK in self.K.nodes}

        # glue together the graphs L\K and R\K along the vertices in K
        self.S = nx.compose(self.L, self.RK)
        self.S.add_edges_from(E_K2RK2K)

        # since the GRR's L-term has been modified, rebuild the seeker
        self.seeker = Seeker(self.L)

    @staticmethod
    def core(HI, new_scope):
        JI = copy.copy(HI)  # rescoping rules implement one-to-one mappings
        JI = nx.relabel_nodes(JI, {n: 'new_' + n for n in JI.nodes})
        nx.set_node_attributes(JI, {n: new_scope for n in JI.nodes if (JI.nodes[n]['bipartite'] == __KERNEL_PARTITION__)}, 'scope')
        return JI

    def apply(self, G, g, new_scope):

        # generate replacement graph (`G.subgraph({v for v in g.values()})` is the match graph 'H')
        I = G.subgraph({k for k, v in g.items() if v in set(self.K.nodes)})
        HI = G.subgraph({k for k, v in g.items() if v not in set(self.K.nodes)})
        JI = self.core(HI, new_scope)

        # get morphism '\mu_{(J \setminus I) \to (R \setminus K)}'
        JI2RK_morphisms = Seeker(self.RK).get_morphisms(JI)
        assert len(JI2RK_morphisms) == 1
        g_JI2RK = JI2RK_morphisms[0]
        g_RK2JI = {v: k for k, v in g_JI2RK.items()}

        # derive new graph
        G = nx.compose(G, JI)  # add non-interface replacement graph
        for vI in I.nodes:  # glue replacement graph to old graph
            vK = g[vI]
            G.add_edges_from({(vI, g_RK2JI[vRK]) for (_, vRK) in self.F_K2RK[vK]})
            G.add_edges_from({(g_RK2JI[vRK], vI) for (vRK, _) in self.F_RK2K[vK]})
        G.remove_nodes_from(HI.nodes)  # discard non-interface nodes of match graph

        return G

    def multiapply(self, G, gs, new_scope):
        for i, g in enumerate(gs):
            G = self.apply(G, g, new_scope + '_' + str(i))
        return G


class AutoRescopeRule(Rule):

    def __init__(self, L, K):
        super(AutoRescopeRule, self).__init__(L, K)

        # define L-term
        self.L = L
        nx.relabel_nodes(self.L, {n: '/'.join(['L-term', n]) for n in set(self.L.nodes).difference(set(K.nodes))}, copy=False)

        # define K-term
        nx.relabel_nodes(self.L, {n: '/'.join(['K-term', n]) for n in set(self.L.nodes).intersection(set(K.nodes))}, copy=False)
        self.K = self.L.subgraph({n for n in set(self.L.nodes) if n.startswith('K-term')})

        # get the graph L\K
        self.LK = self.L.subgraph(set(self.L.nodes).difference({n for n in set(self.L.nodes) if n.startswith('K-term')}))
        # get the edges between the vertices of K and the vertices of L\K
        E_K2LK2K = set(e for e in set(self.L.edges).difference(set(self.LK.edges) | set(self.K.edges)))
        E_K2LK = set(e for e in E_K2LK2K if e[0] in self.K.nodes)
        E_LK2K = set(e for e in E_K2LK2K if e[1] in self.K.nodes)

        # define the graph R\K -- rescoping rules implement one-to-one mappings
        self.RK = nx.relabel_nodes(copy.copy(self.LK), {n: n.replace('L-term', 'R-term') for n in set(self.LK.nodes)})
        # define the edges between the vertices of K and the vertices of R\K
        LK2RK_morphisms = Seeker(self.RK).get_morphisms(self.LK)
        assert len(LK2RK_morphisms) == 1
        g_LK2RK = LK2RK_morphisms[0]
        E_K2RK = set((u, g_LK2RK[v]) for u, v in E_K2LK)
        E_RK2K = set((g_LK2RK[u], v) for u, v in E_LK2K)
        E_K2RK2K = E_K2RK | E_RK2K
        # disintegrate `E_K2RK` and `E_RK2K` along fibres to speed up rule application
        self.F_K2RK = {vK: set(e for e in E_K2RK if e[0] == vK) for vK in self.K.nodes}
        self.F_RK2K = {vK: set(e for e in E_RK2K if e[1] == vK) for vK in self.K.nodes}

        # glue together the graphs L\K and R\K along the vertices in K
        self.S = nx.compose(self.L, self.RK)
        self.S.add_edges_from(E_K2RK2K)

        # since the GRR's L-term has been modified, rebuild the seeker
        self.seeker = Seeker(self.L)

    @staticmethod
    def core(HI):

        # automatically detect the scope of the operations involved (should be unique!)
        scopes = set(HI.nodes[n]['scope'] for n in HI.nodes if (HI.nodes[n]['bipartite'] == __KERNEL_PARTITION__))
        try:
            scopes.remove('')
        except KeyError:
            pass
        assert len(scopes) == 1  # up to now, quantlib's `nn.Module`s traces have included at least one correctly scoped operation...
        new_scope = list(scopes)[0]

        JI = copy.copy(HI)
        JI = nx.relabel_nodes(JI, {n: 'new_' + n for n in JI.nodes})
        nx.set_node_attributes(JI, {n: new_scope for n in JI.nodes if (JI.nodes[n]['bipartite'] == __KERNEL_PARTITION__)}, 'scope')

        return JI

    def apply(self, G, g):

        # generate non-interface replacement graph (`G.subgraph({v for v in g.values()})` is the match graph 'H')
        I = G.subgraph({k for k, v in g.items() if v in set(self.K.nodes)})
        HI = G.subgraph({k for k, v in g.items() if v not in set(self.K.nodes)})
        JI = self.core(HI)

        # get morphism '\mu_{(J \setminus I) \to (R \setminus K)}'
        JI2RK_morphisms = Seeker(self.RK).get_morphisms(JI)
        assert len(JI2RK_morphisms) == 1
        g_JI2RK = JI2RK_morphisms[0]
        g_RK2JI = {v: k for k, v in g_JI2RK.items()}

        # derive new graph
        G = nx.compose(G, JI)  # add non-interface replacement graph
        for vI in I.nodes:  # glue replacement graph to old graph
            vK = g[vI]
            G.add_edges_from({(vI, g_RK2JI[vRK]) for (_, vRK) in self.F_K2RK[vK]})
            G.add_edges_from({(g_RK2JI[vRK], vI) for (vRK, _) in self.F_RK2K[vK]})
        G.remove_nodes_from(HI.nodes)  # detach non-interface match graph

        return G

    def multiapply(self, G, gs):
        for g in gs:
            G = self.apply(G, g)
        return G
