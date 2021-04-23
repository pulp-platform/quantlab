from collections import namedtuple
from networkx.algorithms import bipartite
import tempfile
from datetime import datetime

import quantlib.graphs.graphs
import quantlib.graphs.utils


__FAILURE__ = False
__SUCCESS__ = True


Commit = namedtuple('Commit', ['rho', 'g', 'Gprime', 'nodes_dict'])


class History(object):

    def __init__(self, nx_graph, nodes_dict):
        self._nx_graph = nx_graph  # keep track of the original object
        self._nodes_dict = nodes_dict
        self._undo = []
        self._redo = []

    def show(self):
        print("-- History --")
        for i, commit in enumerate(self._undo):
            print(i, commit)

    def push(self, commit):
        self._undo.append(commit)
        self._redo.clear()

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


class Editor(object):

    def __init__(self, qlgraph, onlykernel=False, graphviz=False):

        self.qlgraph = qlgraph

        if onlykernel:
            G = bipartite.projected_graph(self.qlgraph.nx_graph, {n for n in self.qlgraph.nx_graph.nodes if self.qlgraph.nx_graph.nodes[n]['bipartite'] == quantlib.graphs.graphs.__KERNEL_PARTITION__})
        else:
            G = self.qlgraph.nx_graph
        nodes_dict = {k: v for k, v in self.qlgraph.nodes_dict.items() if k in G.nodes}

        self._history = History(G, nodes_dict)
        self._in_session = False  # puts a lock on the history by preventing editing actions
        self._rho = None  # current GRR
        self._graphviz = graphviz
        self._cache_dir = None

    @property
    def G(self):
        try:
            G = self._history._undo[-1].Gprime
        except IndexError:
            G = self._history._nx_graph
        return G

    @property
    def nodes_dict(self):
        try:
            nodes_dict = self._history._undo[-1].nodes_dict
        except IndexError:
            nodes_dict = self._history._nodes_dict
        return nodes_dict

    def startup(self):
        self._cache_dir = tempfile.TemporaryDirectory()
        import os
        print("Temporary cache directory created at {}".format(os.path.abspath(self._cache_dir.name)))
        self._in_session = True

    def pause(self):
        self._in_session = False

    def resume(self):
        self._in_session = True

    def shutdown(self):
        self._in_session = False
        self._apply_changes_to_graph()
        self._cache_dir.cleanup()
        self._history.clear(force=True)

    def set_grr(self, rho):
        self._rho = rho

    def seek(self, **kwargs):

        if self._rho:
            gs = self._rho.seek(self.G, self.nodes_dict, **kwargs)
        else:
            gs = None
            print("No rule defined.")

        return gs

    def edit(self, gs=None, **kwargs):

        if self._rho and self._in_session:

            if gs is None:
                gs = self.seek(**kwargs)

            for g in gs:

                try:
                    G_new, nodes_dict_new = self._rho.apply(self.G, self.nodes_dict, g)  # derivation
                    self._history.push(Commit(self._rho, g, G_new, nodes_dict_new))
                    status = __SUCCESS__

                except Exception:
                    print("An issue arose while applying rule {} to graph <{}> at point: ".format(type(self._rho), self.G))
                    for vH, vL in g.items():
                        print("\t", vH, vL)
                    status = __FAILURE__

                if (status == __SUCCESS__) and self._graphviz:
                    self._take_snapshot()

        else:
            if self._rho is None:
                print("No rule defined for editor object <{}>.".format(self))
            else:
                print("Editor object <{}> is not in an editing session.".format(self))

    def _apply_changes_to_graph(self):

        self.qlgraph.nx_graph = self.G
        self.qlgraph.nodes_dict = self.nodes_dict

    def _take_snapshot(self):
        filename = datetime.now().strftime("%H:%M:%S_{}_{}".format(len(self._history._undo), type(self._history._undo[-1].rho)))
        quantlib.graphs.utils.draw_graph(self.G, self._cache_dir.name, filename)  # take a snapshot of the edited graph

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


# # exec(open('converter_twn.py').read())
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
