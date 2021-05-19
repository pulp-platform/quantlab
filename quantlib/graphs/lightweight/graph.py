import torch

from .node import LightweightNode

from typing import List


class LightweightGraph(object):

    def __init__(self, net: torch.nn.Module):

        self._net        = net
        self._nodes_list = LightweightGraph.build_nodes_list(self._net, nodes_list=[])

    @property
    def net(self) -> torch.nn.Module:
        return self._net

    @property
    def nodes_list(self) -> List[LightweightNode]:
        return self._nodes_list

    @staticmethod
    def build_nodes_list(parent_module: torch.nn.Module, parent_name: str = '', nodes_list: List[LightweightNode] = []):

        for name, child in parent_module.named_children():
            if len(list(child.children())) == 0:
                nodes_list.append(LightweightNode(name=parent_name + name, module=child))
            else:
                LightweightGraph.build_nodes_list(child, parent_name=parent_name + name + '.', nodes_list=nodes_list)

        return nodes_list

    def rebuild_nodes_list(self):
        self._nodes_list = LightweightGraph.build_nodes_list(self._net, nodes_list=[])

    def show_nodes_list(self):

        for lwn in self._nodes_list:
            print("{:30s} {}".format(lwn.name, lwn.type_))
