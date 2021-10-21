# 
# taskstatistic.py
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

from manager.meter import TaskStatistic
import manager

from manager.platform import PlatformManager
from manager.meter import WriterStub
from typing import Union, Callable, List


class ILSVRC12Statistic(TaskStatistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub,
                 n_epochs: int, n_batches: int,
                 train: bool,
                 postprocess_gt_fun: Callable[[torch.Tensor], torch.Tensor],
                 postprocess_pr_fun: Callable[[Union[torch.Tensor, List[torch.Tensor]]], torch.Tensor]):  # consider the possibility of deep supervision

        name = "Accuracy_TOP{:d}"
        super(ILSVRC12Statistic, self).__init__(platform=platform, writerstub=writerstub,
                                                n_epochs=n_epochs, n_batches=n_batches, name=name,
                                                train=train)

        self._total_tracked = None
        self._total_top1    = None
        self._total_top5    = None
        self._value_top1    = None
        self._value_top5    = None

        self._postprocess_gt_fun = postprocess_gt_fun
        self._postprocess_pr_fun = postprocess_pr_fun

    def _reset(self):
        self._total_tracked = torch.Tensor([0]).to(dtype=torch.int64, device=self._platform.device)
        self._total_top1    = torch.Tensor([0]).to(dtype=torch.int64, device=self._platform.device)
        self._total_top5    = torch.Tensor([0]).to(dtype=torch.int64, device=self._platform.device)
        self._value_top1    = torch.Tensor([0.0]).to(device=self._platform.device)
        self._value_top5    = torch.Tensor([0.0]).to(device=self._platform.device)

    def _stop_observing(self, *args):
        # master-only point: at the end of the epoch, write the running statistics to disk
        if (not self._platform.is_horovod_run) or self._platform.is_master:
            try:
                self._writer.add_scalar(self._tag.format(1), self._value_top1, global_step=self._epoch_id)
                self._writer.add_scalar(self._tag.format(5), self._value_top5, global_step=self._epoch_id)
            except AttributeError:  # ``SummaryWriter`` has not been instantiated
                pass

    def update(self, ygt: torch.Tensor, ypr: torch.Tensor):

        # adapt the ground truth topology labels and the network predictions to the topology-agnostic statistic
        pp_ygt = self._postprocess_gt_fun(ygt)
        pp_ypr = self._postprocess_pr_fun(ypr)

        # compute the batch statistics for the current process
        bs          = torch.Tensor([ypr.shape[0]]).to(dtype=torch.int64, device=ypr.device)
        pp_ypr_top5 = torch.topk(pp_ypr, 5, dim=1).indices
        correct     = pp_ygt == pp_ypr_top5

        n_correct_top1 = torch.sum(correct[:, 0])
        n_correct_top5 = torch.sum(correct)

        # master-workers synchronisation point: different processes apply the model to different data, hence they observe different statistics
        if self._platform.is_multiproc_horovod_run:
            sum_bs             = self._platform.hvd.allreduce(bs,             op=self._platform.hvd.Sum, name='/'.join([self._tag, 'bs']))
            sum_n_correct_top1 = self._platform.hvd.allreduce(n_correct_top1, op=self._platform.hvd.Sum, name=self._tag.format(1))
            sum_n_correct_top5 = self._platform.hvd.allreduce(n_correct_top5, op=self._platform.hvd.Sum, name=self._tag.format(5))
        else:
            sum_bs             = bs
            sum_n_correct_top1 = n_correct_top1
            sum_n_correct_top5 = n_correct_top5

        # update running statistics
        self._total_tracked += sum_bs
        self._total_top1    += sum_n_correct_top1
        self._total_top5    += sum_n_correct_top5
        self._value_top1     = (100.0 * self._total_top1) / self._total_tracked
        self._value_top5     = (100.0 * self._total_top5) / self._total_tracked

        # master-only point: print the running statistic to screen
        if (not self._platform.is_horovod_run) or self._platform.is_master:
            message = manager.QUANTLAB_PREFIX + "Epoch [{:3d}/{:3d}] : Batch [{:5d}/{:5d}] - Accuracy TOP1: {:6.2f}%, Accuracy TOP5: {:6.2f}%".format(self._epoch_id, self._n_epochs, self._batch_id, self._n_batches, self._value_top1.item(), self._value_top5.item())
            print(message)

