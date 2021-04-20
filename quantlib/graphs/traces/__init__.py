import os
from collections import OrderedDict
import networkx as nx

from .trace import __TRACES_LIBRARY__
import quantlib.graphs.graphs


def load_traces_library(modules=None):

    mod_2_trace_dir = {}
    for root, dirs, files in os.walk(__TRACES_LIBRARY__):
        if len(dirs) == 0:  # terminal directories contain only trace files (graphviz, networkx)
            mod_2_trace_dir[os.path.basename(root)] = root

    if modules is None:
        modules = list(mod_2_trace_dir.keys())  # beware: there is no guarantee on the order in which the rescoping rules will be returned!

    libtraces = OrderedDict()
    for mod_name in modules:

        L = nx.read_gpickle(os.path.join(mod_2_trace_dir[mod_name], 'networkx'))
        VK = {n for n in L.nodes if L.nodes[n]['partition'] == quantlib.graphs.graphs.__CONTXT_PARTITION__}
        for n in L.nodes:
            del L.nodes[n]['partition']
        K = L.subgraph(VK)

        libtraces[mod_name] = (L, K)

    return libtraces
