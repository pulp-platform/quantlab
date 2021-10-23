# 
# lossstatistic.py
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

from .statistics import RunningCallbackFreeStatistic
import manager

from ..writerstub import WriterStub
from manager.platform import PlatformManager


class LossStatistic(RunningCallbackFreeStatistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub,
                 n_epochs: int, n_batches: int,
                 train: bool):

        tag = "/".join(["Loss", "Train" if train else "Valid"])
        super(LossStatistic, self).__init__(platform=platform, writerstub=writerstub, tag=tag,
                                            n_epochs=n_epochs, n_batches=n_batches)

        self._total_tracked = None
        self._total_loss    = None
        self._value         = None

    @property
    def value(self):
        return self._value

    def _reset(self):
        self._total_tracked = torch.Tensor([0]).to(dtype=torch.int64)
        self._total_loss    = torch.Tensor([0.0])

    def _stop_observing(self, *args):
        # master-only point: at the end of the epoch, write the running statistic to disk
        if (not self._platform.is_horovod_run) or self._platform.is_master:
            try:
                self._writer.add_scalar(self._tag, self._value, global_step=self._epoch_id)
            except AttributeError:  # ``SummaryWriter`` has not been instantiated
                pass

    def update(self, bs: int, loss: torch.Tensor):

        # compute the batch statistics for the current process
        bs   = torch.Tensor([bs]).to(dtype=torch.int64)
        loss = torch.Tensor([loss.item()])
        loss *= bs  # I assume that loss :obj:`torch.nn.Module`s take the average over the points in the batch

        # master-workers synchronisation point: different processes apply the model to different data, hence they observe different statistics
        if self._platform.is_multiproc_horovod_run:
            sum_bs   = self._platform.hvd.allreduce(bs,   op=self._platform.hvd.Sum, name='/'.join([self._tag, 'bs']))
            sum_loss = self._platform.hvd.allreduce(loss, op=self._platform.hvd.Sum, name='/'.join([self._tag, 'loss']))
        else:
            sum_bs   = bs
            sum_loss = loss

        # update running statistics
        self._total_tracked += sum_bs
        self._total_loss    += sum_loss
        self._value          = self._total_loss / self._total_tracked

        # master-only point: print the running statistic to screen
        if (not self._platform.is_horovod_run) or self._platform.is_master:
            message = manager.QUANTLAB_PREFIX + "Epoch [{:3d}/{:3d}] : Batch [{:5d}/{:5d}] - {}: {:10.3f}".format(self._epoch_id, self._n_epochs, self._batch_id, self._n_batches, self._tag, self._value.item())
            print(message)

