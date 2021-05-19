import re

from ..node import LightweightNode
from typing import List


__all__ = [
    'OrFilter',
    'AndFilter',
    'NameFilter',
    'TypeFilter',
]


class Filter(object):

    def __init__(self, *args):
        pass

    def find(self, nodes_list: List[LightweightNode]) -> List[LightweightNode]:
        raise NotImplementedError

    def __call__(self, nodes_list: List[LightweightNode]) -> List[LightweightNode]:
        return self.find(nodes_list)

    def __and__(self, other):
        return AndFilter(self, other)

    def __or__(self, other):
        return OrFilter(self, other)


class OrFilter(Filter):

    def __init__(self, filter_a: Filter, filter_b: Filter):
        super(OrFilter, self).__init__()
        self._filter_a = filter_a
        self._filter_b = filter_b

    def find(self, nodes_list: List[LightweightNode]) -> List[LightweightNode]:

        filter_a_nodes = self._filter_a(nodes_list)
        filter_b_nodes = self._filter_b(nodes_list)
        return list(set(filter_a_nodes + filter_b_nodes))  # remove duplicates

    def __repr__(self):
        return "".join(["(", repr(self._filter_a), " | ", repr(self._filter_b), ")"])


class AndFilter(Filter):

    def __init__(self, filter_a: Filter, filter_b: Filter):
        super(AndFilter, self).__init__()
        self._filter_a = filter_a
        self._filter_b = filter_b

    def find(self, nodes_list: List[LightweightNode]) -> List[LightweightNode]:

        filter_a_nodes = self._filter_a(nodes_list)
        filter_b_nodes = self._filter_b(filter_a_nodes)

        return filter_b_nodes

    def __repr__(self):
        return "".join(["(", repr(self._filter_a), " & ", repr(self._filter_b), ")"])


class NameFilter(Filter):

    def __init__(self, regex: str):
        super(NameFilter, self).__init__()
        self._regex   = regex
        self._pattern = re.compile(self._regex)

    def find(self, nodes_list: List[LightweightNode]) -> List[LightweightNode]:
        return list(filter(lambda n: self._pattern.match(n.name), nodes_list))

    def __repr__(self):
        return "".join([self.__class__.__name__, "('", self._regex, "')"])


class TypeFilter(Filter):

    def __init__(self, type_: type):
        super(TypeFilter, self).__init__()
        self._type = type_

    def find(self, nodes_list: List[LightweightNode]) -> List[LightweightNode]:
        return list(filter(lambda n: n.type_ == self._type, nodes_list))

    @property
    def _type_str(self):
        return str(self._type).replace("<class '", "").replace("'>", "")

    def __repr__(self):
        return "".join([self.__class__.__name__, "(", self._type_str, ")"])
