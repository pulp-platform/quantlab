# 
# library.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2021 ETH Zurich.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

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

