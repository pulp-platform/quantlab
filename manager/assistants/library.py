import os

import types


class QuantLabLibrary(object):

    def __init__(self, module: types.ModuleType):
        self._module = module

    @property
    def module(self):
        return self._module

    @property
    def name(self):
        return os.path.basename(os.path.dirname(self._module.__file__))

    @property
    def package(self):
        return self._module.__package__
