from collections import namedtuple
import networkx as nx
import graphviz as gv
import torch.nn as nn

import quantlib.graphs.graphs


__all__ = [
    'draw_graph',
]


GVNodeAppearance = namedtuple('GVNodeAppearance', ['fontsize', 'shape', 'height', 'width', 'color', 'fillcolor'])


__KERNEL_PARTITION__ = quantlib.graphs.graphs.__KERNEL_PARTITION__
__MEMORY_PARTITION__ = quantlib.graphs.graphs.__MEMORY_PARTITION__
__CONTXT_PARTITION__ = quantlib.graphs.graphs.__CONTXT_PARTITION__


_styles = {
    __KERNEL_PARTITION__: GVNodeAppearance(fontsize='8', shape='circle', height='2.0', width='2.0',
                                           color='cornflowerblue', fillcolor='cornflowerblue'),
    __MEMORY_PARTITION__: GVNodeAppearance(fontsize='8', shape='square', height='1.2', width='1.2',
                                           color='brown2', fillcolor='brown2'),
    __CONTXT_PARTITION__: GVNodeAppearance(fontsize='8', shape='square', height='1.2', width='1.2',
                                           color='chartreuse', fillcolor='chartreuse')
}


def draw_graph(G, dir, node_names=None, name='graphviz'):

    gvG = gv.Digraph(comment=name)

    for n, p in nx.get_node_attributes(G, 'bipartite').items():
        gvG.node(n, node_names[n] if node_names else G.nodes[n]['type'], **_styles[p]._asdict(), style='filled')

    for e in G.edges:
        gvG.edge(e[0], e[1])

    gvG.render(directory=dir, filename=name)


class ViewFlattenNd(nn.Module):

    def __init__(self):
        super(ViewFlattenNd, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
