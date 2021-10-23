# 
# learningratestatistic.py
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

from .statistics import InstantaneousCallbackFreeStatistic

from ..writerstub import WriterStub
from manager.platform import PlatformManager


__all__ = [
    'LearningRateStatistic',
]


class LearningRateStatistic(InstantaneousCallbackFreeStatistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub,
                 n_epochs: int, n_batches: int,
                 opt: torch.optim.Optimizer, writer_kwargs: dict = {}):

        tag = "Learning_rate"
        super(LearningRateStatistic, self).__init__(platform=platform, writerstub=writerstub, tag=tag,
                                                    n_epochs=n_epochs, n_batches=n_batches,
                                                    start=0, period=n_batches)

        self._opt = opt
        self._writer_kwargs = writer_kwargs

    def _start_observing(self):
        self._writer.add_scalar(self._tag, self._opt.param_groups[0]['lr'], global_step=self._epoch_id, **self._writer_kwargs)

    def _stop_observing(self):
        pass

