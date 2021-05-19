from collections import namedtuple
import torch

from ..node import LightweightNode
from ..graph import LightweightGraph
from .filters import Filter
from typing import List, Callable


Application = namedtuple('Application', ['path', 'old_module', 'new_module'])


class LightweightRule(object):

    def __init__(self, filter_: Filter, replacement_fun: Callable[[torch.nn.Module], torch.nn.Module]):

        self._filter          = filter_
        self._replacement_fun = replacement_fun

    @property
    def filter(self):
        return self._filter

    @staticmethod
    def get_module(parent_module: torch.nn.Module, path_to_target: List[str]) -> torch.nn.Module:
        """Return a handle on the target ``torch.nn.Module``.

        When we want to replace the target module with a quantized
        counterpart, the retrieved module can be used to extract the structural
        parameters that need to be passed to the constructor method of the
        quantized module (e.g., the kernel size for convolutional layers).
        """

        if len(path_to_target) == 1:
            module = parent_module._modules[path_to_target[0]]
        else:
            module = LightweightRule.get_module(parent_module._modules[path_to_target[0]], path_to_target[1:])

        return module

    @staticmethod
    def replace_module(parent_module: torch.nn.Module, path_to_target: List[str], new_module: torch.nn.Module):
        """Replace the target ``torch.nn.Module`` with a given counterpart.

        This function is mostly meant to be used to replace floating-point
        PyTorch modules with their fake-quantized counterparts implemented in
        the ``quantlib.algorithms`` sub-package.

        Side effect: the state of the  ``parent_module`` is changed by this
        operation, since one or more of its children sub-modules are replaced.
        """

        if len(path_to_target) == 1:
            parent_module._modules[path_to_target[0]] = new_module
        else:
            LightweightRule.replace_module(parent_module._modules[path_to_target[0]], path_to_target[1:], new_module)

    def _seek(self, nodes_list: List[LightweightNode]) -> List[LightweightNode]:
        return self._filter(nodes_list)

    def apply(self, graph: LightweightGraph) -> List[Application]:

        applications = []

        for lwn in self._seek(graph.nodes_list):

            old_module = self.get_module(graph.net, lwn.path)
            new_module = self._replacement_fun(old_module)

            self.replace_module(graph.net, lwn.path, new_module)

            applications.append(Application(path=lwn.path, old_module=old_module, new_module=new_module))

        graph.rebuild_nodes_list()

        return applications

    def unapply(self, graph: LightweightGraph, applications: List[Application]) -> None:

        for a in applications:
            self.replace_module(graph.net, a.path, a.old_module)

        graph.rebuild_nodes_list()
