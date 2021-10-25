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

from .statistics import RunningCallbackFreeStatistic

from ..writerstub import WriterStub
from manager.platform import PlatformManager
from typing import Union, List


class TaskStatistic(RunningCallbackFreeStatistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub,
                 n_epochs: int, n_batches: int,
                 name: str, train: bool):

        tag = "/".join([name, "Train" if train else "Valid"])
        super(TaskStatistic, self).__init__(platform=platform, writerstub=writerstub, tag=tag,
                                            n_epochs=n_epochs, n_batches=n_batches)

    def _reset(self):
        raise NotImplementedError

    def _stop_observing(self, *args):
        raise NotImplementedError

    def update(self, ygt: torch.Tensor, ypr: Union[torch.Tensor, List[torch.Tensor]]):  # consider the possibility of deep supervision
        raise NotImplementedError

