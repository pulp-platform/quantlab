# 
# visionstatistics.py
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

from .statistics import InstantaneousCallbackBasedStatistic

from ..writerstub import WriterStub
from manager.platform import PlatformManager
from typing import Tuple, Union


__all__ = [
    'OutputFeaturesSnapshot',
]


_RGB_01_NORMALIZED       = 0
_RGB_DATA_SET_NORMALIZED = 1
_RGB_UINT8               = 2


class RGBInputsSnapshot(InstantaneousCallbackBasedStatistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub,
                 n_epochs: int, n_batches: int,
                 start: int, period: int,
                 name: str, module: torch.nn.Module, n_inputs: int, preprocessing_type: int, preprocessing_stats: dict, writer_kwargs: dict = {}):

        tag = "/".join(["Input_image", name])
        super(RGBInputsSnapshot, self).__init__(platform=platform, writerstub=writerstub, tag=tag,
                                                n_epochs=n_epochs, n_batches=n_batches,
                                                start=start, period=period,
                                                module=module)

        assert isinstance(self._module, torch.nn.Conv2d)
        assert self._module.in_channels == 3

        self._n_inputs = n_inputs

        self._preprocessing_type = preprocessing_type
        self._mean = torch.Tensor(preprocessing_stats['mean']).unsqueeze(-1).unsqueeze(-1).to(self._platform.device)
        self._std  = torch.Tensor(preprocessing_stats['std']).unsqueeze(-1).unsqueeze(-1).to(self._platform.device)

        self._writer_kwargs = writer_kwargs

    def _hook_fn(self, module: torch.nn.Module, inputs: Union[Tuple[torch.Tensor, ...], torch.Tensor], outputs: Union[Tuple[torch.Tensor, ...], torch.Tensor]):

        if (not self._platform.is_horovod_run) or self._platform.is_master:

            if isinstance(inputs, tuple):
                input_ = inputs[0]
            else:
                input_ = inputs

            if self._preprocessing_type == _RGB_01_NORMALIZED:
                pass
            elif self._preprocessing_type == _RGB_DATA_SET_NORMALIZED:
                denormalised_input = input_ * self._std + self._mean
            elif self._preprocessing_type == _RGB_UINT8:
                denormalised_input = input_ / 255.0  # map to the range [0, 1]

            for input_id in range(0, self._n_inputs):
                try:
                    self._writer.add_image("/".join([self._tag, str(input_id)]), denormalised_input[input_id], global_step=self._global_step, **self._writer_kwargs)
                except AttributeError:  # ``SummaryWriter`` has not been instantiated
                    pass

    def _start_observing(self):
        self._handle = self._module.register_forward_hook(self._hook_fn)

    def _stop_observing(self):
        self._handle.remove()


class OutputFeaturesSnapshot(InstantaneousCallbackBasedStatistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub,
                 n_epochs: int, n_batches: int, start: int, period: int,
                 name: str, module: torch.nn.Module, n_inputs: int, writer_kwargs: dict = {}):

        tag = "/".join(["Output_features", name])
        super(OutputFeaturesSnapshot, self).__init__(platform=platform, writerstub=writerstub, tag=tag,
                                                     n_epochs=n_epochs, n_batches=n_batches,
                                                     start=start, period=period,
                                                     module=module)

        self._n_inputs = n_inputs
        self._writer_kwargs = writer_kwargs

    def _hook_fn(self, module: torch.nn.Module, inputs: Union[Tuple[torch.Tensor, ...], torch.Tensor], outputs: Union[Tuple[torch.Tensor, ...], torch.Tensor]):

        if (not self._platform.is_horovod_run) or self._platform.is_master:

            if isinstance(outputs, tuple):
                output_ = outputs[0]
            else:
                output_ = outputs
            assert output_.dim() == 4  # batch, channel, height, width

            denormalised_output_ = (output_ - torch.min(output_)) / (torch.max(output_) - torch.min(output_))  # map to the range [0, 1]

            for input_id in range(0, self._n_inputs):
                try:
                    self._writer.add_images("/".join([self._tag, str(input_id)]), denormalised_output_[input_id].unsqueeze(1), global_step=self._global_step, **self._writer_kwargs)
                except AttributeError:  # ``SummaryWriter`` has not been instantiated
                    pass

    def _start_observing(self):
        self._handle = self._module.register_forward_hook(self._hook_fn)

    def _stop_observing(self):
        self._handle.remove()

