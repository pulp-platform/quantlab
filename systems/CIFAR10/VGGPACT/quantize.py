# 
# quantize.py
# 
# Author(s):
# Georg Rutishauser <georgr@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2021 ETH Zurich. All rights reserved.
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

from torch import nn

from quantlib.algorithms.pact import PACTUnsignedAct, PACTConv2d, PACTLinear
from quantlib.algorithms.pact import PACTActController, PACTLinearController
from quantlib.editing.lightweight import LightweightGraph, LightweightEditor
from quantlib.editing.lightweight.rules import LightweightRule
from quantlib.editing.lightweight.rules.filters import TypeFilter, VariadicOrFilter


class PACTSequential(LightweightGraph):

    def __init__(self, net: nn.Module, config: dict):

        super(PACTSequential, self).__init__(net)
        # make filter for convs
        conv_filter = VariadicOrFilter(*[TypeFilter(t) for t in (nn.Conv2d, PACTConv2d)])
        self.conv_filter = conv_filter
        # make filter for linears
        lin_filter = VariadicOrFilter(*[TypeFilter(t) for t in (nn.Linear, PACTLinear)])
        self.lin_filter = lin_filter
        # make filter for activations
        act_filter = VariadicOrFilter(*[TypeFilter(t) for t in (nn.ReLU, PACTUnsignedAct)])
        self.act_filter = act_filter
        if config is not None:
            self.conv_config = config['PACTConv2d']
            self.lin_config = config['PACTLinear']
            self.act_config = config['PACTUnsignedAct']
        else:
            self.conv_config = None
            self.lin_config = None
            self.act_config = None

    @property
    def conv_nodes(self):
        return self.conv_filter.find(self.nodes_list)

    @property
    def linear_nodes(self):
        return self.lin_filter.find(self.nodes_list)

    @property
    def act_nodes(self):
        return self.act_filter.find(self.nodes_list)

    def quantize_convs(self):
        # n_levels=-1 is interpreted as "do not quantize"
        # conv_config['n_levels'] should be a key-value dict with indices into
        # the list of conv nodes as keys and the number of levels as values
        if self.conv_config is None:
            print("PACTSequential: quantize_convs() called but no config supplied - returning!")
            return
        lvl_cfg = {int(k): v for k, v in self.conv_config['n_levels'].items() if v > 0}
        # all other config entries should be usable as kwargs for PACTConv2d
        other_cfg = {k: v for k, v in self.conv_config.items() if k != 'n_levels'}

        def replace_conv(c: nn.Conv2d, n_levels: int):
            pc = PACTConv2d.from_conv2d(c, n_levels=n_levels, **other_cfg)
            return pc

        for k, n in lvl_cfg.items():
            node_to_replace = self.conv_nodes[k]
            pc = replace_conv(node_to_replace.module, n)
            LightweightRule.replace_module(self.net, node_to_replace.path, pc)

    def quantize_lins(self):
        if self.lin_config is None:
            print("PACTSequential: quantize_lins() called but no config supplied - returning!")
            return
        lvl_cfg = {int(k): v for k, v in self.lin_config['n_levels'].items() if v > 0}
        other_cfg = {k: v for k, v in self.lin_config.items() if k != 'n_levels'}

        def replace_lin(l: nn.Linear, n_levels: int):
            pc = PACTLinear.from_linear(l, n_levels=n_levels, **other_cfg)
            return pc

        for k, n in lvl_cfg.items():
            node_to_replace = self.linear_nodes[k]
            pl = replace_lin(node_to_replace.module, n)
            LightweightRule.replace_module(self.net, node_to_replace.path, pl)

    def quantize_acts(self):
        if self.act_config is None:
            print("PACTSequential: quantize_acts() called but no config supplied - returning!")
            return
        lvl_cfg = {int(k): v for k, v in self.act_config['n_levels'].items() if v > 0}
        other_cfg = {k: v for k, v in self.act_config.items() if k != 'n_levels'}
        for k, n in lvl_cfg.items():
            node_to_replace = self.act_nodes[k]
            pa = PACTUnsignedAct(n_levels=n, **other_cfg)
            LightweightRule.replace_module(self.net, node_to_replace.path, pa)

    def quantize(self):
        self.quantize_convs()
        self.quantize_lins()
        self.quantize_acts()
        self.rebuild_nodes_list()

    def get_lin_controller(self, schedule: dict, verbose: bool = False):
        lin_modules = [n.module for n in self.conv_nodes + self.linear_nodes if type(n) in [PACTConv2d, PACTLinear] ]
        return PACTLinearController(lin_modules, schedule, verbose=verbose)

    def get_act_controller(self, schedule: dict, verbose: bool = False):
        act_modules = [n.module for n in self.act_nodes if isinstance(n, PACTUnsignedAct)]
        return PACTActController(act_modules, schedule, verbose=verbose)


def quantize_pact(net, config):
    g = PACTSequential(net, config)
    g.quantize()
    return g.net


def get_pact_controllers(net, schedules, verbose=False):
    g = PACTSequential(net, None)
    return [g.get_lin_controller(schedule=schedules['linear'], verbose=verbose), g.get_act_controller(schedule=schedules['activation'], verbose=verbose)]

