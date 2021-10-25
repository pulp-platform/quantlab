# 
# writerstub.py
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

from torch.utils.tensorboard import SummaryWriter

from typing import Union


class WriterStub(object):

    def __init__(self):
        self._writer = None

    @property
    def writer(self) -> Union[SummaryWriter, None]:
        return self._writer

    @writer.setter
    def writer(self, summarywriter: SummaryWriter) -> None:
        self._writer = summarywriter

