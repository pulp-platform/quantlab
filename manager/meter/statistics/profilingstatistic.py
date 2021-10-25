# 
# profilingstatistic.py
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
import time

from .statistics import InstantaneousCallbackFreeStatistic

from ..writerstub import WriterStub
from manager.platform import PlatformManager


class ProfilingStatistic(InstantaneousCallbackFreeStatistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub,
                 n_epochs: int, n_batches: int):

        tag = "Points-per-second"
        super(ProfilingStatistic, self).__init__(platform=platform, writerstub=writerstub, tag=tag,
                                                 n_epochs=n_epochs, n_batches=n_batches,
                                                 start=0, period=1)

        self._tic = None
        self._toc = None

        self._value = None

    def _start_observing(self):
        self._tic = time.time()

    def _stop_observing(self, bs: int):

        self._toc = time.time()
        elapsed   = self._toc - self._tic

        # cast to PyTorch tensors (this is mandatory when using Horovod communication API)
        bs      = torch.Tensor([bs]).to(dtype=torch.int64)
        elapsed = torch.Tensor([elapsed])

        # master-workser synchronisation point: different processes might take different amounts of time to process the same amount of data, even if the workload is in principle equally distributed
        if self._platform.is_multiproc_horovod_run:
            sum_bs      = self._platform.hvd.allreduce(bs, op=self._platform.hvd.Sum, name='/'.join([self._tag, 'bs']))
            max_elapsed = torch.max(self._platform.hvd.allgather(elapsed, name='/'.join([self._tag, 'elapsed'])))  # why does not Horovod support an allreduce max operator?
        else:
            sum_bs      = bs
            max_elapsed = elapsed

        # compute the statistic (number of data points processed per second)
        self._value = sum_bs / max_elapsed

        # master-only point: write the statistic to disk
        if (not self._platform.is_horovod_run) or self._platform.is_master:
            try:
                self._writer.add_scalar(self._tag, self._value, global_step=self._global_step)
            except AttributeError:  # ``SummaryWriter`` has not been instantiated
                pass

    def maybe_stop_observing(self, bs: int):
        if self._is_observing:
            self._stop_observing(bs)
            self._is_observing = False

