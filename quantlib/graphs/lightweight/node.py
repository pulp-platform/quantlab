import torch

from typing import List


class LightweightNode(object):

    def __init__(self, name: str, module: torch.nn.Module):
        self.name   = name
        self.module = module

    @property
    def path(self) -> List[str]:
        return self.name.split('.')

    @property
    def type_(self) -> type:
        return type(self.module)
