# 
# taskstatistic.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
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

import torch

from manager.meter import TaskStatistic
import manager

from manager.platform import PlatformManager
from manager.meter import WriterStub
from typing import Union, Callable, List


class DVS128Statistic(TaskStatistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub,
                 n_epochs: int, n_batches: int,
                 train: bool,
                 postprocess_gt_fun: Callable[[torch.Tensor], torch.Tensor],
                 postprocess_pr_fun: Callable[[Union[torch.Tensor, List[torch.Tensor]]], torch.Tensor]):  # consider the possibility of deep supervision

        name = "Accuracy"
        super(DVS128Statistic, self).__init__(platform=platform, writerstub=writerstub,
                                               n_epochs=n_epochs, n_batches=n_batches, name=name,
                                               train=train)

        self._total_tracked = None
        self._total_correct = None
        self._value         = None

        self._postprocess_gt_fun = postprocess_gt_fun
        self._postprocess_pr_fun = postprocess_pr_fun

    def _reset(self):
        self._total_tracked = torch.Tensor([0]).to(dtype=torch.int64, device=self._platform.device)
        self._total_correct = torch.Tensor([0]).to(dtype=torch.int64, device=self._platform.device)
        self._value         = torch.Tensor([0.0]).to(device=self._platform.device)

    def _stop_observing(self, *args):
        # master-only point: at the end of the epoch, write the running statistics to disk
        if (not self._platform.is_horovod_run) or self._platform.is_master:
            try:
                self._writer.add_scalar(self._tag, self._value, global_step=self._epoch_id)
            except AttributeError:  # ``SummaryWriter`` has not been instantiated
                pass

    def update(self, ygt: torch.Tensor, ypr: torch.Tensor):

        # adapt the ground truth topology labels and the network predictions to the topology-agnostic statistic
        pp_ygt = self._postprocess_gt_fun(ygt)
        pp_ypr = self._postprocess_pr_fun(ypr)

        # compute the batch statistics for the current process
        bs        = torch.Tensor([ypr.shape[0]]).to(dtype=torch.int64, device=self._platform.device)
        correct   = pp_ygt == pp_ypr
        n_correct = torch.sum(correct)

        # master-workers synchronisation point: different processes apply the model to different data, hence they observe different statistics
        if self._platform.is_multiproc_horovod_run:
            sum_bs        = self._platform.hvd.allreduce(bs,        op=self._platform.hvd.Sum, name='/'.join([self._tag, 'bs']))
            sum_n_correct = self._platform.hvd.allreduce(n_correct, op=self._platform.hvd.Sum, name=self._tag)
        else:
            sum_bs        = bs
            sum_n_correct = n_correct

        # update running statistics
        self._total_tracked += sum_bs
        self._total_correct += sum_n_correct
        self._value          = (100.0 * self._total_correct) / self._total_tracked

        # master-only point: print the running statistic to screen
        if (not self._platform.is_horovod_run) or self._platform.is_master:
            message = manager.QUANTLAB_PREFIX + "Epoch [{:3d}/{:3d}] : Batch [{:5d}/{:5d}] - Accuracy: {:6.2f}%".format(self._epoch_id, self._n_epochs, self._batch_id, self._n_batches, self._value.item())
            print(message)

