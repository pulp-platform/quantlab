# 
# training.py
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

from collections import namedtuple
import importlib
from operator import attrgetter
from typing import Union, List
import torch

import systems.utils

from .library import QuantLabLibrary
from manager.platform import PlatformManager
from quantlib.algorithms import Controller


GradientDescent = namedtuple('GradientDescent', ['opt', 'lr_sched'])  # an update algorithm for gradient descent consists of two parts: computing the updates given the gradients and (possibly) dynamically updating the step length hyper-parameter


class TrainingMessage(object):

    def __init__(self, config: dict, library: QuantLabLibrary) -> None:
        """Describe how to set and solve the optimisation (learning) problem.

        An object of this class implements the server-side of a *dependency
        injection* pattern whose client-side is a :obj:`TrainingAssistant`
        object.

        Args:
            config: the functional description of the loss function, the
                gradient descent algorithm (including the optimiser and
                possibly the learning rate scheduler), and (possibly) the
                schedules of the quantization algorithms.
            library: the collection of class and function definitions that
                allow to assemble a loss function (if custom), an optimisation
                algorithm if custom, (possibly) a learning rate scheduler (if
                custom), and (possibly) the collection of quantization
                ``Controller``s.

        """

        self._config  = config
        self._library = library

    @property
    def config(self):
        return self._config

    @property
    def library(self):
        return self._library


class TrainingAssistant(object):

    def __init__(self):
        """The object that assembles the optimisation (learning) problem.

        An object of this class implements the client-side of a **dependency
        injection* pattern whose server-side is a :obj:`TrainingMessage`
        object. QuantLab assumes that ``TrainingMessage``s are created by a
        :obj:`Logbook` instance according to the machine learning system's
        library and on the experimental unit's configuration.

        This class follows a **builder** design pattern.
        """

        self._loss_fn_class     = None
        self._loss_fn_kwargs    = None
        self._loss_fn_takes_net = False  # non-standard, topology-dependent loss function

        self._opt_class     = None
        self._opt_kwargs    = None
        self._opt_takes_net = False  # non-standard, topology-dependent optimiser

        self._lr_sched_class  = None
        self._lr_sched_kwargs = None

        self._qnt_ctrls_fun    = None
        self._qnt_ctrls_kwargs = None

    def recv_trainingmessage(self, trainingmessage: TrainingMessage) -> None:
        """Resolve the functional dependencies for the assembly.

        Args:
            trainingmessage: the collection of dependencies that the
                ``NetworkAssistant`` should be aware of when assembling the
                components required to train the target deep neural network.

        """

        # loss function (mandatory)
        loss_fn_getter = attrgetter(trainingmessage.config['loss_fn']['class'])
        if 'takes_net' in trainingmessage.config['loss_fn'].keys():
            self._loss_fn_takes_net = trainingmessage.config['loss_fn']['takes_net']

        try:
            self._loss_fn_class = loss_fn_getter(torch.nn)
        except AttributeError:
            # the loss function is custom
            # search order (from less to more problem/topology specific):
            #   1. systems library
            #   2. topology library
            try:
                self._loss_fn_class = loss_fn_getter(systems.utils.loss)
            except AttributeError:
                self._loss_fn_class = loss_fn_getter(trainingmessage.library.module)
        self._loss_fn_kwargs = trainingmessage.config['loss_fn']['kwargs']

        # optimisation algorithm - optimiser (mandatory)
        opt_getter = attrgetter(trainingmessage.config['gd']['opt']['class'])
        if 'takes_net' in trainingmessage.config['gd']['opt'].keys():
            self._opt_takes_net = trainingmessage.config['gd']['opt']['takes_net']

        try:
            self._opt_class = opt_getter(torch.optim)
        except AttributeError:
            # the optimiser is custom
            # search order (from less to more problem/topology specific):
            #   1. systems library
            #   2. topology library
            try:
                self._opt_class = opt_getter(systems.utils.optimizer)
            except AttributeError:
                self._opt_class = opt_getter(trainingmessage.library.module)
        self._opt_kwargs = trainingmessage.config['gd']['opt']['kwargs']

        # optimisation algorithm - learning rate scheduler (optional)
        if 'lr_sched' in trainingmessage.config['gd'].keys():
            lr_sched_getter = attrgetter(trainingmessage.config['gd']['lr_sched']['class'])
            try:
                self._lr_sched_class = lr_sched_getter(torch.optim.lr_scheduler)
            except AttributeError:
                # the learning rate scheduler is custom
                # search order (from less to more problem/topology specific):
                #   1. systems library
                #   2. topology library
                try:
                    self._lr_sched_class = lr_sched_getter(systems.utils.lrscheduler)
                except AttributeError:
                    self._lr_sched_class = lr_sched_getter(trainingmessage.library.module)
            self._lr_sched_kwargs = trainingmessage.config['gd']['lr_sched']['kwargs']

        # quantization controllers (optional)
        if ('quantize' in trainingmessage.config.keys()) and (trainingmessage.config['quantize'] is not None):
            qnt_library = importlib.import_module('.quantize', package=trainingmessage.library.package)
            self._qnt_ctrls_fun    = getattr(qnt_library, trainingmessage.config['quantize']['function'])
            self._qnt_ctrls_kwargs = trainingmessage.config['quantize']['kwargs']

    def prepare_loss(self, net: torch.nn.Module) -> torch.nn.Module:
        loss_fn = self._loss_fn_class(net, **self._loss_fn_kwargs) if self._loss_fn_takes_net else self._loss_fn_class(**self._loss_fn_kwargs)
        return loss_fn

    def prepare_gd(self, platform: PlatformManager, net: torch.nn.Module) -> GradientDescent:

        # optimiser
        opt = self._opt_class(net, **self._opt_kwargs) if self._opt_takes_net else self._opt_class(net.parameters(), **self._opt_kwargs)
        if platform.is_horovod_run:
            opt = platform.hvd.DistributedOptimizer(opt, named_parameters=net.named_parameters())

        # learning rate scheduler
        lr_sched = self._lr_sched_class(opt, **self._lr_sched_kwargs) if self._lr_sched_class is not None else None

        gd = GradientDescent(opt=opt, lr_sched=lr_sched)
        return gd

    def prepare_qnt_ctrls(self, net: torch.nn.Module) -> List[Controller]:
        qnt_ctrls = self._qnt_ctrls_fun(net, **self._qnt_ctrls_kwargs) if self._qnt_ctrls_fun is not None else []
        return qnt_ctrls

    # def prepare(self, platform: PlatformManager, net: torch.nn.Module) -> Tuple[torch.nn.Module, GradientDescent, Union[List[object], List]]:
    #     """Create the objects required to perform optimisation.
    #
    #     Args:
    #         platform: the entity that registers the engineering aspects of the
    #             computation: hardware specifications, OS details, MPI
    #             configuration (via Horovod).
    #         net: the target deep neural network.
    #
    #     Returns:
    #         (tuple):
    #
    #             loss_fn: the loss function.
    #             gd: the gradient descent algorithm; its functions are
    #                 computing the actual updates and (possibly) dynamically
    #                  updating the step length parameter (i.e., the learning
    #                  rate).
    #             qnt_ctrls: the controllers for the hyper-parameters that
    #                 govern the quantization algorithms.
    #
    #     """
    #
    #     loss_fn   = self.prepare_loss()
    #     gd        = self.prepare_gd(platform, net)
    #     qnt_ctrls = self.prepare_qnt_ctrls(net)
    #
    #     return loss_fn, gd, qnt_ctrls
