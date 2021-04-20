import torch
import torch.nn
import networkx as nx

from quantlib.graphs.grrules.dporules import DPORule


class FoldINQConv2dSTEARule(DPORule):

    def __init__(self):

        self.L = nx.DiGraph()
