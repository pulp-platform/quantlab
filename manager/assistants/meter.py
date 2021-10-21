# 
# meter.py
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

from collections import namedtuple, OrderedDict
import torch

import manager.meter
from manager.meter import Meter
from manager.meter import WriterStub

from .library import QuantLabLibrary
from manager.platform import PlatformManager
from typing import Union, List, Dict


StatisticDescription = namedtuple('StatisticDescription', ['class_', 'kwargs'])


class MeterMessage(object):

    def __init__(self, n_epochs: int, config: dict, tlibrary: QuantLabLibrary, plibrary: QuantLabLibrary, writerstub_epoch: WriterStub, writerstub_step: WriterStub) -> None:

        self._n_epochs = n_epochs
        self._config   = config
        self._tlibrary = tlibrary
        self._plibrary = plibrary

        self._writerstub_epoch = writerstub_epoch
        self._writerstub_step  = writerstub_step

    @property
    def n_epochs(self):
        return self._n_epochs

    @property
    def config(self):
        return self._config

    @property
    def tlibrary(self):
        return self._tlibrary

    @property
    def plibrary(self):
        return self._plibrary

    @property
    def writerstub_epoch(self):
        return self._writerstub_epoch

    @property
    def writerstub_step(self):
        return self._writerstub_step


class MeterAssistant(object):

    def __init__(self, partition: str):

        self._partition = partition

        self._n_epochs                       = None

        self._task_statistic_class           = None
        self._compute_task_statistic         = False
        self._tensor_statistics_descriptions = []
        self._compute_profiling_statistic    = False

        self._writerstub_epoch = None
        self._writerstub_step  = None

    def _resolve_tensor_statistics_configurations(self, config: List[Dict], tlibrary: QuantLabLibrary, plibrary: QuantLabLibrary) -> None:

        for statistic_config in config:

            # hierarchy of resolution is: 1. topology library, 2. problem library, 3. manager library
            try:
                class_ = getattr(tlibrary.module, statistic_config['class'])
            except AttributeError:
                try:
                    class_ = getattr(plibrary.module, statistic_config['class'])
                except AttributeError:
                    class_ = getattr(manager.meter, statistic_config['class'])

            kwargs = statistic_config['kwargs']
            self._tensor_statistics_descriptions.append(StatisticDescription(class_=class_, kwargs=kwargs))

    def recv_metermessage(self, metermessage):

        self._n_epochs = metermessage.n_epochs

        # task statistic
        self._task_statistic_class = getattr(metermessage.tlibrary.module, ''.join([metermessage.tlibrary.name, 'Statistic']))  # the name of the topology-specific task statistic is prefixed by the topology name
        if self._partition == 'train':
            self._compute_task_statistic = metermessage.config['compute_task_statistic']
        else:
            self._compute_task_statistic = True

        # tensor statistics (optional)
        try:
            config = metermessage.config['tensor_statistics']
            self._resolve_tensor_statistics_configurations(config, metermessage.tlibrary, metermessage.plibrary)
        except KeyError:  # no tensor statistics defined
            pass

        # compute profiling statistic? (mandatory)
        self._compute_profiling_statistic = metermessage.config['compute_profiling_statistic']

        # writer stubs
        self._writerstub_epoch = metermessage.writerstub_epoch
        self._writerstub_step  = metermessage.writerstub_step

    @staticmethod
    def _names_2_modules(module: torch.nn.Module, parent_name: str = '', n2m: OrderedDict = OrderedDict()):
        """Compute a mapping of ``Module`` names to ``Module`` objects."""

        for n, m in module.named_children():
            if len(list(m.children())) == 0:  # ``Module`` is not ``nn.Sequential`` or other container type; i.e., this is a "leaf" ``Module``
                n2m.update({parent_name + n: m})
            else:
                MeterAssistant._names_2_modules(m, parent_name=''.join([parent_name, n, '.']), n2m=n2m)

        return n2m

    def _assemble_meter(self,
                        platform: PlatformManager,
                        n_batches: int,
                        net: torch.nn.Module,
                        opt: Union[torch.optim.Optimizer, None] = None) -> Meter:

        meter = Meter(self._n_epochs, n_batches)

        # loss statistic (always computed)
        meter.register_statistic(manager.meter.LossStatistic(platform=platform, writerstub=self._writerstub_epoch, n_epochs=meter.n_epochs, n_batches=meter.n_batches, train=True if self._partition == 'train' else False))

        # task statistic (optional)
        if self._compute_task_statistic:
            meter.register_statistic(self._task_statistic_class(platform=platform, writerstub=self._writerstub_epoch, n_epochs=meter.n_epochs, n_batches=meter.n_batches, train=True if self._partition == 'train' else False))

        # learning rate statistic (always computed)
        if self._partition == 'train':
            assert opt is not None
            meter.register_statistic(manager.meter.LearningRateStatistic(platform=platform, writerstub=self._writerstub_epoch, n_epochs=meter.n_epochs, n_batches=meter.n_batches, opt=opt))

        # tensor statistics
        n2m = self._names_2_modules(net)  # tensor statistics are bound to :obj:`torch.nn.Module`s by name; computing this mapping once avoids looping multiple times through the network to discover the binding points
        for sd in self._tensor_statistics_descriptions:
            name = '.'.join(['module', sd.kwargs['name']]) if platform.is_nndataparallel_run else sd.kwargs['name']  # `torch.nn.DataParallel` wraps network objects adding a naming layer
            sd.kwargs['module'] = n2m[name]  # resolve module name into ``torch.nn.Module`` object
            meter.register_statistic(sd.class_(platform=platform, writerstub=self._writerstub_step, n_epochs=meter.n_epochs, n_batches=meter.n_batches, **sd.kwargs))

        # profiling statistic (optional)
        if self._compute_profiling_statistic:
            meter.register_statistic(manager.meter.ProfilingStatistic(platform=platform, writerstub=self._writerstub_step, n_epochs=meter.n_epochs, n_batches=meter.n_batches))

        return meter

    def prepare(self,
                platform: PlatformManager,
                n_batches: int,
                net: torch.nn.Module,
                opt: Union[torch.optim.Optimizer, None] = None) -> Meter:

        meter = self._assemble_meter(platform=platform, n_batches=n_batches, net=net, opt=opt)

        return meter

