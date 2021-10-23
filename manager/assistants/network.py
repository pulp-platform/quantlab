# 
# network.py
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

import importlib
import torch.nn as nn

from .library import QuantLabLibrary
from manager.platform import PlatformManager
from typing import Union, Callable


class NetworkMessage(object):

    def __init__(self, config: dict, library: QuantLabLibrary) -> None:
        """Describe how to build a :obj:`torch.nn.Module`.

        An object of this class implements the server-side of a *dependency
        injection* pattern whose client-side is a :obj:`NetworkAssistant`
        object.

        Args:
            config: the functional description of the network topology,
                (possibly) a quantization recipe, and (possibly) initialise
                it.
            library: the collection of class and function definitions that
                allow to assemble an ``nn.Module``, (possibly) quantize it,
                and (possibly) load parameters from a pre-trained model (i.e.,
                assigning an arbitrary initial condition for the experiment).

        """

        self._config  = config
        self._library = library

    @property
    def config(self):
        return self._config

    @property
    def library(self):
        return self._library


class NetworkAssistant(object):

    def __init__(self):
        """The entity that assembles :obj:`torch.nn.Module``s.

        An object of this class implements the client-side of a *dependency
        injection* pattern whose server-side is a :obj:`NetworkMessage`
        object. QuantLab assumes that ``NetworkMessage``s are created by a
        :obj:`Logbook` instance according to the machine learning system's
        library and on the experimental unit's configuration.

        This class follows a *factory* pattern.

        Attributes:
            _network_class (Callable[..., torch.nn.Module]): the class
                describing the network topology to instantiate, or the family
                of network topologies from which to instantiate a specific
                topology.
            _network_kwargs (dict): the keyword arguments that specify how to
                instantiate the network object.
            _network seeds (list): the seeds for the initialisation algorithm
                of the ``nn.Module``; one seed per fold.
            _qnt_recipe_fun (Callable[..., torch.nn.Module]): the function
                describing how to convert a full-precision network into a
                fake-quantized network; the first argument of such function
                should be an ``nn.Module`` object.
            _qnt_recipe_kwargs (dict): the keyword arguments that specify how
                to apply the quantization recipe (e.g., the formats of the
                layers' operands).
            _pretrained_fun (Callable[..., None]): the function describing how
                to load parameters from a pre-trained ``nn.Module`` into the
                newly created ``nn.Module``.

        """

        self._network_class  = None
        self._network_kwargs = None
        self._network_seeds  = None

        self._qnt_recipe_fun    = None
        self._qnt_recipe_kwargs = None

        # TODO: support loading a pre-trained model (in it's essence, this is the initial condition of a dynamical system)
        # self._pretrained_fun    = None
        # self._pretrained_kwargs = None

    def recv_networkmessage(self, networkmessage: NetworkMessage) -> None:
        """Resolve the functional dependencies for the assembly.

        Args:
            networkmessage: the collection of dependencies that the
                ``NetworkAssistant`` should be aware of when building the
                ``nn.Module``.

        """

        # deep neural network (mandatory)
        self._network_class  = getattr(networkmessage.library.module, networkmessage.config['class'])
        self._network_kwargs = networkmessage.config['kwargs']
        self._network_seeds  = networkmessage.config['seeds']

        # quantization recipe (optional)
        if ('quantize' in networkmessage.config.keys()) and (networkmessage.config['quantize'] is not None):
            qnt_library = importlib.import_module('.quantize', package=networkmessage.library.package)
            self._qnt_recipe_fun    = getattr(qnt_library, networkmessage.config['quantize']['function'])
            self._qnt_recipe_kwargs = networkmessage.config['quantize']['kwargs']

        # TODO: import pre-trained model loading function

    def prepare(self, platform: PlatformManager, fold_id: int) -> Union[nn.Module, nn.DataParallel]:
        """Create the ``nn.Module``.

        Args:
            platform: the entity that registers the engineering aspects of the
                computation: hardware specifications, OS details, MPI
                configuration (via Horovod).
            fold_id: the identifier of the fold for which to prepare the
                ``nn.Module``.

        Returns:
            net: the deep neural network that will be trained or tested; in
                non-Horovod, single-process, multi-GPU runs, the ``nn.Module``
                is wrapped inside an :obj:`torch.nn.DataParallel` object to
                speed up computations.

        """

        net = self._network_class(**self._network_kwargs, seed=self._network_seeds[fold_id])  # WARNING: all QuantLab networks require a seed now!

        if self._qnt_recipe_fun:
            net = self._qnt_recipe_fun(net, **self._qnt_recipe_kwargs)

        # TODO: apply pre-trained model loading function

        net = net.to(platform.device)

        if platform.is_nndataparallel_run:
            net = nn.DataParallel(net)  # single-node, single-process, multi-GPU run

        # master-workers synchronisation point: if the network has been initialised from file, then the master should communicate the network parameters to the workers
        if platform.is_multiproc_horovod_run:
            platform.hvd.broadcast_parameters(net.state_dict(), root_rank=platform.master_rank)

        return net
