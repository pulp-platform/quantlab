from collections import namedtuple
import importlib
import torch

from .library import QuantLabLibrary
from manager.platform import PlatformManager
from typing import Union, List, Tuple


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

        Training deep neural networks

        """

        self._loss_fn_class  = None
        self._loss_fn_kwargs = None

        self._opt_class  = None
        self._opt_kwargs = None

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
        try:
            self._loss_fn_class = getattr(torch.nn, trainingmessage.config['loss_fn']['class'])
        except AttributeError:  # the loss function is custom
            self._loss_fn_class = getattr(trainingmessage.library.module, trainingmessage.config['loss_fn']['class'])
        self._loss_fn_kwargs = trainingmessage.config['loss_fn']['kwargs']

        # optimisation algorithm - optimiser (mandatory)
        self._opt_class = getattr(torch.optim, trainingmessage.config['gd']['opt']['class'])
        self._opt_kwargs = trainingmessage.config['gd']['opt']['kwargs']

        # optimisation algorithm - learning rate scheduler (optional)
        if 'lr_sched' in trainingmessage.config['gd'].keys():
            try:
                self._lr_sched_class = getattr(torch.optim.lr_scheduler, trainingmessage.config['gd']['lr_sched']['class'])
            except AttributeError:  # the learning rate scheduler is custom
                self._lr_sched_class = getattr(trainingmessage.library.module, trainingmessage.config['gd']['lr_sched']['kwargs'])
            self._lr_sched_kwargs = trainingmessage.config['gd']['lr_sched']['kwargs']

        # quantization controllers (optional)
        if 'quantize' in trainingmessage.config.keys():
            qnt_library = importlib.import_module('.'.join('', 'quantize'), package=trainingmessage.library.module)
            self._qnt_ctrls_fun    = getattr(qnt_library, trainingmessage.config['quantize']['function'])
            self._qnt_ctrls_kwargs = trainingmessage.config['quantize']['kwargs']

    def get_loss(self) -> torch.nn.Module:
        loss_fn = self._loss_fn_class(**self._loss_fn_kwargs)
        return loss_fn

    def get_gd(self, platform: PlatformManager, net: torch.nn.Module) -> GradientDescent:

        opt = self._opt_class(net.parameters(), **self._opt_kwargs)
        if platform.is_horovod_run:
            opt = platform.hvd.DistributedOptimizer(opt, named_parameters=net.named_parameters())

        lr_sched = self._lr_sched_class(opt, **self._lr_sched_kwargs) if self._lr_sched_class is not None else None

        gd = GradientDescent(opt=opt, lr_sched=lr_sched)
        return gd

    def get_qnt_ctrls(self, net: torch.nn.Module) -> Union[List[object], List]:  # TODO: return list of QuantLab ``Controller``s
        qnt_ctrls = self._qnt_ctrls_fun(net, **self._qnt_ctrls_kwargs) if self._qnt_ctrls_fun is not None else []
        return qnt_ctrls

    def prepare(self, platform: PlatformManager, net: torch.nn.Module) -> Tuple[torch.nn.Module, GradientDescent, Union[List[object], List]]:
        """Create the objects required to perform optimisation.

        Args:
            platform: the entity that registers the engineering aspects of the
                computation: hardware specifications, OS details, MPI
                configuration (via Horovod).
            net: the target deep neural network.

        Returns:
            (tuple):

                loss_fn: the loss function.
                gd: the gradient descent algorithm; its functions are
                    computing the actual updates and (possibly) dynamically
                     updating the step length parameter (i.e., the learning
                     rate).
                qnt_ctrls: the controllers for the hyper-parameters that
                    govern the quantization algorithms.

        """

        loss_fn   = self.get_loss()
        gd        = self.get_gd(platform, net)
        qnt_ctrls = self.get_qnt_ctrls(net)

        return loss_fn, gd, qnt_ctrls
