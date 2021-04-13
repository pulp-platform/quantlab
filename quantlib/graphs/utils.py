from collections import namedtuple
import networkx as nx
import graphviz as gv

from quantlib.graphs.morph import __KERNEL_PARTITION__, __MEMORY_PARTITION__, __CONTXT_PARTITION__

__all__ = [
    'draw_graph',
]


GVNodeAppearance = namedtuple('GVNodeAppearance', ['fontsize', 'shape', 'height', 'width', 'color', 'fillcolor'])


_styles = {
    __KERNEL_PARTITION__: GVNodeAppearance(fontsize='8', shape='circle', height='2.0', width='2.0',
                                           color='cornflowerblue', fillcolor='cornflowerblue'),
    __MEMORY_PARTITION__: GVNodeAppearance(fontsize='8', shape='square', height='1.2', width='1.2', color='brown2',
                                           fillcolor='brown2'),
    __CONTXT_PARTITION__: GVNodeAppearance(fontsize='8', shape='square', height='1.2', width='1.2',
                                           color='chartreuse', fillcolor='chartreuse')
}


def draw_graph(G, dir, name='graphviz'):

    gvG = gv.Digraph(comment=name)

    for n, p in nx.get_node_attributes(G, 'partition').items():
        gvG.node(n, "\n".join([G.nodes[n]['scope'], G.nodes[n]['type']]), **_styles[p]._asdict())

    for e in G.edges:
        gvG.edge(e[0], e[1])

    gvG.render(directory=dir, filename=name)
