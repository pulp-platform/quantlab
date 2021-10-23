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

import torch

from .statistics.statistics            import TimeStatistic
from .statistics.profilingstatistic    import ProfilingStatistic
from .statistics.lossstatistic         import LossStatistic
from .statistics.taskstatistic         import TaskStatistic
from .statistics.learningratestatistic import LearningRateStatistic
from .statistics.statistics            import InstantaneousStatistic


class Meter(object):

    def __init__(self, n_epochs: int, n_batches: int) -> None:

        self._n_epochs    = n_epochs
        self._n_batches   = n_batches
        self._global_step = None

        self._loss_statistic      = None
        self._task_statistic      = None
        self._lr_statistic        = None
        self._tensor_statistics   = []
        self._profiling_statistic = None

        self._best_loss = float('inf')
        self._is_best   = None

    @property
    def n_epochs(self) -> int:
        return self._n_epochs

    @property
    def epoch_id(self) -> int:
        return self._global_step // self._n_batches

    @property
    def n_batches(self) -> int:
        return self._n_batches

    @property
    def batch_id(self) -> int:
        return self._global_step % self._n_batches

    def register_statistic(self, s: TimeStatistic) -> None:

        if isinstance(s, LossStatistic):
            self._loss_statistic = s
        elif isinstance(s, TaskStatistic):
            self._task_statistic = s
        elif isinstance(s, LearningRateStatistic):
            self._lr_statistic = s
        elif isinstance(s, InstantaneousStatistic) and not isinstance(s, ProfilingStatistic):
            self._tensor_statistics.append(s)
        elif isinstance(s, ProfilingStatistic):
            self._profiling_statistic = s

    def step(self, epoch_id: int, batch_id: int) -> None:
        self._global_step = epoch_id * self.n_batches + batch_id

    def start_observing(self):

        # running statistics
        # loss
        self._loss_statistic.maybe_start_observing(self._global_step)
        # task
        if self._task_statistic is not None:
            self._task_statistic.maybe_start_observing(self._global_step)

        # learning rate statistic
        if self._lr_statistic is not None:
            self._lr_statistic.maybe_start_observing(self._global_step)

        # tensor statistics - hook callbacks to ``torch.nn.Module``s
        for s in self._tensor_statistics:
            s.maybe_start_observing(self._global_step)

    def tic(self):
        """Start profiling."""
        if self._profiling_statistic is not None:
            self._profiling_statistic.maybe_start_observing(self._global_step)

    def update(self, ygt: torch.Tensor, ypr: torch.Tensor, loss: torch.Tensor) -> None:
        """Update running statistics."""
        # loss
        bs = ygt.shape[0]  # batch size
        self._loss_statistic.update(bs, loss)
        # task
        if self._task_statistic is not None:
            self._task_statistic.update(ygt, ypr)

    def toc(self, ygt: torch.Tensor) -> None:
        """Stop profiling."""
        bs = ygt.shape[0]
        if self._profiling_statistic is not None:
            self._profiling_statistic.maybe_stop_observing(bs)

    def stop_observing(self):

        # running statistics
        # loss
        self._loss_statistic.maybe_stop_observing()
        # task
        if self._task_statistic is not None:
            self._task_statistic.maybe_stop_observing()

        # learning rate statistic
        if self._lr_statistic is not None:
            self._lr_statistic.maybe_stop_observing()

        # tensor statistics - release hooks placed on ``torch.nn.Module``s
        for s in self._tensor_statistics:
            s.maybe_stop_observing()

    @property
    def best_loss(self) -> float:
        return self._best_loss

    @best_loss.setter
    def best_loss(self, value):
        self._best_loss = value

    @property
    def is_best(self) -> bool:
        return self._is_best

    def check_improvement(self):

        if self._loss_statistic.value.item() < self.best_loss:
            self._best_loss = self._loss_statistic.value.item()
            self._is_best   = True
        else:
            self._is_best = False
