# 
# taskstatistic.py
# 
# Author(s):
# Philip Wiese <wiesep@student.ethz.ch>
# 
# Copyright (c) 2023 ETH Zurich.
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

from systems.CIFAR10.utils.statistics import CIFAR10Statistic

from manager.platform import PlatformManager
from manager.meter import WriterStub


def _postprocess_gt(ygt: torch.Tensor) -> torch.Tensor:
    return ygt.unsqueeze(-1)


def _postprocess_pr(ypr: torch.Tensor) -> torch.Tensor:
    return ypr.argmax(dim=1).unsqueeze(-1)


class CCTStatistic(CIFAR10Statistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub,
                 n_epochs: int, n_batches: int,
                 train: bool):
        super(CCTStatistic, self).__init__(platform=platform, writerstub=writerstub,
                                           n_epochs=n_epochs, n_batches=n_batches,
                                           train=train,
                                           postprocess_gt_fun=_postprocess_gt, postprocess_pr_fun=_postprocess_pr)

