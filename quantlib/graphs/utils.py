import networkx as nx
import graphviz as gv
from collections import namedtuple
import os

from quantlib.graphs.morph import __KERNEL_PARTITION__, __MEMORY_PARTITION__


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


GVNodeAppearance = namedtuple('GVNodeAppearance', ['fontsize', 'shape', 'height', 'width', 'color', 'fillcolor'])


__APP_DICT__ = {
    __KERNEL_PARTITION__: GVNodeAppearance(fontsize='8', shape='circle', height='2.0', width='2.0',
                                           color='cornflowerblue', fillcolor='cornflowerblue'),
    __MEMORY_PARTITION__: GVNodeAppearance(fontsize='8', shape='square', height='1.2', width='1.2', color='brown2',
                                           fillcolor='brown2')
}


def draw_graph(G, dir, name='graphviz'):

    gvG = gv.Digraph(comment=name)

    for n, p in nx.get_node_attributes(G, 'bipartite').items():
        gvG.node(n, G.nodes[n]['type'],
                 fontsize=__APP_DICT__[p].fontsize,
                 shape=__APP_DICT__[p].shape,
                 fixedsize='true', height=__APP_DICT__[p].height, width=__APP_DICT__[p].width,
                 color=__APP_DICT__[p].color,
                 style='filled', fillcolor=__APP_DICT__[p].fillcolor)

    for e in G.edges:
        gvG.edge(e[0], e[1])

    gvG.render(directory=dir, filename=name)


def torch_jit_create_label():
    pass


def torch_jit_label_graph():
    pass


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
