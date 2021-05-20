from networkx.algorithms import isomorphism


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
        
        # G is the whole network graph, T is the template to be matched
        matcher = isomorphism.DiGraphMatcher(G, self.T)
        
        # Return list of all occurences of the non-attributed template
        # This means only the number of nodes + interconnections are matched
        isomorphisms = list(matcher.subgraph_isomorphisms_iter())

        # check the second morphism condition (label consistency)
        # This checks that node types / classes (technically, string labels representing the class name) are as in the template
        morphisms = [g for g in isomorphisms if Seeker.is_morphism(self.T, G, g)]

        # remove duplicate morphisms
        # SCHEREMO: Check this a bit carefully
        # SPMATTEO: Check what is filtered, pipe to logs
        unique_VHs = set(frozenset(g.keys()) for g in morphisms)
        VHs_2_morphisms = {VH: [g for g in morphisms if frozenset(g.keys()) == VH] for VH in unique_VHs}
        morphisms = [v[0] for v in VHs_2_morphisms.values()]

        # List of dictionaries, mapping node identifiers to node identifiers
        # Node identifier as in the networkx package
        
        # We map the node identifiers that are absolute/unique to the original graph G
        # to node identifiers that are a bsolute/unique to the template graph self.T
        # This assures there is no ambiguity in the mapping (<- This is the assumption here)
        
        return morphisms
