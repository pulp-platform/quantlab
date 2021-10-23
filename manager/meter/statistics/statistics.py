# 
# statistics.py
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

import torch

from ..writerstub import WriterStub
from manager.platform import PlatformManager


__all__ = [
    'RunningCallbackFreeStatistic',
    'InstantaneousCallbackFreeStatistic',
    'InstantaneousCallbackBasedStatistic',
]


class Statistic(object):

    def __init__(self, platform: PlatformManager, writerstub: WriterStub, tag: str):

        self._platform   = platform
        self._writerstub = writerstub

        self._tag = tag

    @property
    def _writer(self):
        return self._writerstub.writer


class CallbackStatistic(Statistic):
    """Track a statistic using PyTorch's callback system.

    Deep learning frameworks can be partitioned according to the way in which
    they build computational graphs: static-graph and dynamic-graph.
    Static-graph frameworks such as TensorFlow compile the computational graph
    before running the first forward pass, and keep a handle on this structure
    until the end of the scope inside which the graph is needed. Dynamic-graph
    frameworks such as PyTorch instead build the computational graph anew at
    each forward pass, and release the components of the structure to the
    garbage collector as soon as the corresponding backward pass is executed.

    QuantLab is based on PyTorch, whose dynamic-graph strategy makes it more
    complicated to perform some operations, such as graph tracing and
    statistics tracking. In fact, the only :obj:`torch.Tensor`s that can be
    explicitly accessed by name are inputs, outputs, and parameters. The
    access is made possible by the fact that inputs and outputs are usually
    assigned to names in the program's scope, and the :obj:`torch.nn.Module`s
    that compose a network store their parameters in attributes, and the
    ``Module``s themselves can be accessed as attributes of the network object
    (itself assigned to a name in the program's scope).

    Sometimes, it might be interesting to look at the intermediate features of
    a network (when looking for an intuition about the model's internal
    representations) or look at its internally-flowing gradients (when
    debugging the training of a particularly hard-to-train network). Accessing
    such internal data structures is possible in PyTorch with the so-called
    *hooks*. Hooks are callbacks that can be *registered* (i.e., assigned to
    specific ``Module``s) either as *forward hooks* or *backward hooks*. The
    difference between forward and backward hooks is that forward hooks will
    be executed whenever the ``forward`` method of the ``Module`` with which
    the hook has been registered is called, whereas backward hooks will be
    executed when the :obj:`torch.autograd` function associated with the
    forward method is called (hence, backward hooks are never executed when
    the network is executed in a :obj:`torch.no_grad()` context manager).

    QuantLab assumes that the use-cases for callback-based statistics are such
    that it is only interesting to take instantaneous measurements. I.e., no
    accumulations or reductions over multiple iterations of the training,
    validation, or test loops are supported.

    """

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub, tag: str,
                 requires_callback: bool):
        super(CallbackStatistic, self).__init__(platform=platform, writerstub=writerstub, tag=tag)
        self.__requires_callback = requires_callback


class CallbackFreeStatistic(CallbackStatistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub, tag: str):
        super(CallbackFreeStatistic, self).__init__(platform=platform, writerstub=writerstub, tag=tag,
                                                    requires_callback=False)


class CallbackBasedStatistic(CallbackStatistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub, tag: str):#,
                 # module: torch.nn.Module):
        super(CallbackBasedStatistic, self).__init__(platform=platform, writerstub=writerstub, tag=tag,
                                                     requires_callback=True)
        # self._module          = module
        self._callback_handle = None


class TimeStatistic(Statistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub, tag: str,
                 is_instantaneous: bool, n_epochs: int, n_batches: int):

        super(TimeStatistic, self).__init__(platform=platform, writerstub=writerstub, tag=tag)
        self.__is_instantaneous = is_instantaneous

        self._n_epochs    = n_epochs
        self._n_batches   = n_batches
        self._global_step = None

        self._is_observing = False

    @property
    def _epoch_id(self) -> int:
        return self._global_step // self._n_batches

    @property
    def _batch_id(self) -> int:
        return self._global_step % self._n_batches

    def _start_observing(self, *args):
        raise NotImplementedError

    def _stop_observing(self, *args):
        raise NotImplementedError

    def maybe_start_observing(self, *args):
        raise NotImplementedError

    def maybe_stop_observing(self, *args):
        raise NotImplementedError


class RunningStatistic(TimeStatistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub, tag: str,
                 n_epochs: int, n_batches: int):
        super(RunningStatistic, self).__init__(platform=platform, writerstub=writerstub, tag=tag,
                                               is_instantaneous=False, n_epochs=n_epochs, n_batches=n_batches)

    def _reset(self, *args):
        raise NotImplementedError

    def _start_observing(self, *args):
        self._reset(*args)

    def _stop_observing(self, *args):
        raise NotImplementedError

    def maybe_start_observing(self, global_step: int, *args):
        self._global_step = global_step
        if self._batch_id == 0:
            self._is_observing = True
            self._start_observing(*args)  # at the beginning of each epoch, reset the tracking

    def update(self, *args):
        raise NotImplementedError

    def maybe_stop_observing(self, *args):
        assert self._is_observing
        if self._batch_id == self._n_batches - 1:
            self._stop_observing(*args)  # write to disk
            self._is_observing = False


class InstantaneousStatistic(TimeStatistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub, tag: str,
                 n_epochs: int, n_batches: int,
                 start: int, period: int):

        super(InstantaneousStatistic, self).__init__(platform=platform, writerstub=writerstub, tag=tag,
                                                     is_instantaneous=True, n_epochs=n_epochs, n_batches=n_batches)

        self._start  = start
        self._period = period
        assert 0 <= self._start < self._period

    def _start_observing(self, *args):
        raise NotImplementedError

    def _stop_observing(self, *args):
        raise NotImplementedError

    def maybe_start_observing(self, global_step: int, *args):
        self._global_step = global_step
        if self._global_step % self._period == self._start:
            self._is_observing = True
            self._start_observing(*args)

    def maybe_stop_observing(self, *args):
        if self._is_observing:
            self._stop_observing(*args)
            self._is_observing = False


class RunningCallbackFreeStatistic(RunningStatistic, CallbackFreeStatistic):
    """Track and compute a running statistic on the outputs of a network.

    The ultimate goal of training a learning system is attaining satisfying
    performance at the task for which it is intended. By definition of
    learning problem, we do not know the exact distribution of the points that
    the system will process once it is deployed in the real world (the
    so-called *real distribution*). This epistemological problem implies that
    the performance of the system can never be really measured, and we can
    just aim at computing estimates for it. The best shot that we can have at
    such estimate is using the system to process all the points of a data set
    to which we have access. The optimality of this strategy comes from one of
    the fundamental assumptions of supervised learning: i.e., that the
    distribution of the data points in these data sets is very similar to real
    distribution.

    To derive the performance estimate, we just need to observe the outputs of
    the network and compare them with the ground truth; we do not need to look
    at internal statistics of the model being trained. In principle, this
    comparison can be done *a posteriori*, i.e., after all the outputs have
    been computed, supposing all the information has been stored in some file.
    Unfortunately, when the data set contains millions or billions of data
    points and the output data structures are large, storing this information
    in a file can be impractical. Hence, it is better (although more
    complicated) to compute these statistics on a running basis.

    """
    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub, tag: str,
                 n_epochs: int, n_batches: int):
        super().__init__(platform=platform, writerstub=writerstub, tag=tag,
                         n_epochs=n_epochs, n_batches=n_batches)

    def _reset(self, *args):
        raise NotImplementedError

    def _stop_observing(self, *args):
        raise NotImplementedError

    def update(self, *args):
        raise NotImplementedError


class InstantaneousCallbackFreeStatistic(InstantaneousStatistic, CallbackFreeStatistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub, tag: str,
                 n_epochs: int, n_batches: int,
                 start: int, period: int):
        super().__init__(platform=platform, writerstub=writerstub, tag=tag,
                         n_epochs=n_epochs, n_batches=n_batches,
                         start=start, period=period)

    def _start_observing(self, *args):
        raise NotImplementedError

    def _stop_observing(self, *args):
        raise NotImplementedError


class InstantaneousCallbackBasedStatistic(InstantaneousStatistic, CallbackBasedStatistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub, tag: str,
                 n_epochs: int, n_batches: int,
                 start: int, period: int,
                 module: torch.nn.Module):
        super().__init__(platform=platform, writerstub=writerstub, tag=tag,
                         n_epochs=n_epochs, n_batches=n_batches,
                         start=start, period=period)
        # super().__init__(module=module)
        self._module = module

    def _start_observing(self, *args):
        raise NotImplementedError

    def _stop_observing(self, *args):
        raise NotImplementedError

