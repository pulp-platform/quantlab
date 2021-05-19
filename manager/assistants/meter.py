# -*- coding: utf-8 -*-
from collections import namedtuple, OrderedDict
import torch

import manager.meter
from manager.meter import Meter
from manager.meter import WriterStub

from .library import QuantLabLibrary
from manager.platform import PlatformManager
from manager.logbook.logsmanager import LogsManager
from typing import Union, Tuple, List, Dict


StatisticDescription = namedtuple('StatisticDescription', ['class_', 'kwargs'])


class MeterMessage(object):

    def __init__(self, n_epochs: int, config: dict, tlibrary: QuantLabLibrary, plibrary: QuantLabLibrary) -> None:

        self._n_epochs = n_epochs
        self._config   = config
        self._tlibrary = tlibrary
        self._plibrary = plibrary

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


class MeterAssistant(object):

    def __init__(self):

        self._n_epochs = None
        self._task_statistic_class = None

        self._train_profiling_statistic = False
        self._train_task_statistic      = False
        self._train_tensor_statistics   = []

        self._valid_profiling_statistic = False
        self._valid_tensor_statistics   = []

    def _resolve_tensor_statistics_configurations(self, config: List[Dict], tlibrary: QuantLabLibrary, plibrary: QuantLabLibrary, train: bool) -> None:

        tensor_statistics_descriptions = []

        for statistic_config in config:

            try:
                class_ = getattr(tlibrary.module, statistic_config['class'])
            except AttributeError:
                try:
                    class_ = getattr(plibrary.module, statistic_config['class'])
                except AttributeError:
                    class_ = getattr(manager.meter, statistic_config['class'])

            kwargs = statistic_config['kwargs']
            tensor_statistics_descriptions.append(StatisticDescription(class_=class_, kwargs=kwargs))

        if train:
            self._train_tensor_statistics.extend(tensor_statistics_descriptions)
        else:
            self._valid_tensor_statistics.extend(tensor_statistics_descriptions)

    def recv_metermessage(self, metermessage):

        self._n_epochs = metermessage.n_epochs

        task_statistic_class_name  = ''.join([metermessage.tlibrary.name, 'Statistic'])
        self._task_statistic_class = getattr(metermessage.tlibrary.module, task_statistic_class_name)

        # training statistics
        # compute profiling statistic during training? (mandatory)
        self._train_profiling_statistic = metermessage.config['train']['compute_profiling_statistic']
        # compute task statistic during training? (mandatory)
        self._train_task_statistic = metermessage.config['train']['compute_task_statistic']
        # tensor statistics (optional)
        try:
            config = metermessage.config['train']['tensor_statistics']
            self._resolve_tensor_statistics_configurations(config, metermessage.tlibrary, metermessage.plibrary, train=True)
        except KeyError:  # no tensor statistics defined
            pass

        # validation statistics
        # compute profiling statistic during training? (mandatory)
        self._valid_profiling_statistic = metermessage.config['valid']['compute_profiling_statistic']
        # tensor statistics (optional)
        try:
            config = metermessage.config['valid']['tensor_statistics']
            self._resolve_tensor_statistics_configurations(config, metermessage.tlibrary, metermessage.plibrary, train=False)
        except KeyError:  # no tensor statistics defined
            pass

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
                        train: bool,
                        platform: PlatformManager,
                        writerstub_epoch: WriterStub,
                        writerstub_step: WriterStub,
                        n_batches: int,
                        net: torch.nn.Module,
                        opt: Union[torch.optim.Optimizer, None] = None) -> Meter:

        meter = Meter(self._n_epochs, n_batches)

        # profiling statistic (optional)
        if train and self._train_profiling_statistic:
            meter.register_statistic(manager.meter.ProfilingStatistic(platform=platform, writerstub=writerstub_step, n_epochs=meter.n_epochs, n_batches=meter.n_batches))
        elif not train and self._valid_profiling_statistic:
            meter.register_statistic(manager.meter.ProfilingStatistic(platform=platform, writerstub=writerstub_step, n_epochs=meter.n_epochs, n_batches=meter.n_batches))

        # loss statistic (always computed)
        meter.register_statistic(manager.meter.LossStatistic(platform=platform, writerstub=writerstub_epoch, n_epochs=meter.n_epochs, n_batches=meter.n_batches, train=train))

        # task statistic (optional)
        if train and self._train_task_statistic:
            meter.register_statistic(self._task_statistic_class(platform=platform, writerstub=writerstub_epoch, n_epochs=meter.n_epochs, n_batches=meter.n_batches, train=train))
        elif not train:
            meter.register_statistic(self._task_statistic_class(platform=platform, writerstub=writerstub_epoch, n_epochs=meter.n_epochs, n_batches=meter.n_batches, train=train))

        # learning rate statistic (always computed)
        if train:
            assert opt is not None
            meter.register_statistic(manager.meter.LearningRateStatistic(platform=platform, writerstub=writerstub_epoch, n_epochs=meter.n_epochs, n_batches=meter.n_batches, opt=opt))

        # tensor statistics
        n2m = self._names_2_modules(net)  # tensor statistics are bound to :obj:`torch.nn.Module`s by name; computing this mapping once avoids looping multiple times through the network to discover the binding points
        if train:
            for sd in self._train_tensor_statistics:
                name = '.'.join(['module', sd.kwargs['name']]) if platform.is_nndataparallel_run else sd.kwargs['name']  # `torch.nn.DataParallel` wraps network objects adding a naming layer
                sd.kwargs['module'] = n2m[name]  # resolve module name into ``torch.nn.Module`` object
                meter.register_statistic(sd.class_(platform=platform, writerstub=writerstub_step, n_epochs=meter.n_epochs, n_batches=meter.n_batches, **sd.kwargs))
        elif not train:
            for sd in self._valid_tensor_statistics:
                name = '.'.join(['module', sd.kwargs['name']]) if platform.is_nndataparallel_run else sd.kwargs['name']  # `torch.nn.DataParallel` wraps network objects adding a naming layer
                sd.kwargs['module'] = n2m[name]  # resolve module name into ``torch.nn.Module`` object
                meter.register_statistic(sd.class_(platform=platform, writerstub=writerstub_step, n_epochs=meter.n_epochs, n_batches=meter.n_batches, **sd.kwargs))

        return meter

    def prepare(self,
                platform: PlatformManager,
                logs_manager: LogsManager,
                n_batches_train: int,
                n_batches_valid: int,
                net: torch.nn.Module,
                opt: torch.optim.Optimizer) -> Tuple[Meter, Meter]:

        if logs_manager is not None:
            writerstub_epoch      = logs_manager.writerstub_epoch
            writerstub_step_train = logs_manager.writerstub_step_train
            writerstub_step_valid = logs_manager.writerstub_step_valid
        else:  # this code is being executed during a multi-process run, and this branch should be taken by worker processes only (who should not have write permissions to the logs folder)
            writerstub_epoch      = WriterStub()
            writerstub_step_train = WriterStub()
            writerstub_step_valid = WriterStub()

        meter_train = self._assemble_meter(train=True,  platform=platform, writerstub_epoch=writerstub_epoch, writerstub_step=writerstub_step_train, n_batches=n_batches_train, net=net, opt=opt)
        meter_valid = self._assemble_meter(train=False, platform=platform, writerstub_epoch=writerstub_epoch, writerstub_step=writerstub_step_valid, n_batches=n_batches_valid, net=net)

        return meter_train, meter_valid
