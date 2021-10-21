# 
# tensorstatistics.py
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

from typing import Union, Tuple

from .statistics import InstantaneousCallbackBasedStatistic
from .statistics import InstantaneousCallbackFreeStatistic

from ..writerstub import WriterStub
from manager.platform import PlatformManager


__all__ = [
    'DistributionInputFeaturesNorm',
    'DistributionInputFeaturesComponents',
    'DistributionOutputFeaturesNorm',
    'DistributionOutputFeaturesComponents',
    'DistributionIncomingGradientNorm',
    'DistributionIncomingGradientComponents',
    'DistributionOutgoingGradientNorm',
    'DistributionOutgoingGradientComponents',
    'MeanUpdateNormWeightNormRatio',
]


class DistributionInputFeaturesNorm(InstantaneousCallbackBasedStatistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub,
                 n_epochs: int, n_batches: int,
                 start: int, period: int,
                 name: str, module: torch.nn.Module, writer_kwargs: dict = {}):

        tag = "/".join(["Input_features_norm", name])
        super(DistributionInputFeaturesNorm, self).__init__(platform=platform, writerstub=writerstub, tag=tag,
                                                            n_epochs=n_epochs, n_batches=n_batches,
                                                            start=start, period=period,
                                                            module=module)
        self._writer_kwargs = writer_kwargs

    def _hook_fn(self, module: torch.nn.Module, inputs: Union[Tuple[torch.Tensor, ...], torch.Tensor], outputs: Union[Tuple[torch.Tensor, ...], torch.Tensor]):

        if isinstance(inputs, tuple):
            input_ = inputs[0]
        else:
            input_ = inputs

        if self._platform.is_multiproc_horovod_run:
            all_input = self._platform.hvd.allgather(input_, name=self._tag)
        else:
            all_input = input_

        norms = torch.sqrt(torch.sum(torch.pow(all_input, 2), dim=tuple(range(1, all_input.dim()))))

        # master-only point: write the statistic to disk
        if (not self._platform.is_horovod_run) or self._platform.is_master:
            try:
                self._writer.add_histogram(self._tag, norms, global_step=self._global_step, **self._writer_kwargs)
            except AttributeError:  # ``SummaryWriter`` has not been instantiated
                pass

    def _start_observing(self):
        self._handle = self._module.register_forward_hook(self._hook_fn)

    def _stop_observing(self):
        self._handle.remove()


class DistributionInputFeaturesComponents(InstantaneousCallbackBasedStatistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub,
                 n_epochs: int, n_batches: int,
                 start: int, period: int,
                 name: str, module: torch.nn.Module, writer_kwargs: dict = {}):

        tag = "/".join(["Input_features_components", name])
        super(DistributionInputFeaturesComponents, self).__init__(platform=platform, writerstub=writerstub, tag=tag,
                                                                  n_epochs=n_epochs, n_batches=n_batches,
                                                                  start=start, period=period,
                                                                  module=module)
        self._writer_kwargs = writer_kwargs

    def _hook_fn(self, module: torch.nn.Module, inputs: Union[Tuple[torch.Tensor, ...], torch.Tensor], outputs: Union[Tuple[torch.Tensor, ...], torch.Tensor]):

        if isinstance(inputs, tuple):
            input_ = inputs[0]
        else:
            input_ = inputs

        if self._platform.is_multiproc_horovod_run:
            all_input = self._platform.hvd.allgather(input_, name=self._tag)
        else:
            all_input = input_

        # master-only point: write the statistic to disk
        if (not self._platform.is_horovod_run) or self._platform.is_master:
            try:
                self._writer.add_histogram(self._tag, all_input, global_step=self._global_step, **self._writer_kwargs)
            except AttributeError:  # ``SummaryWriter`` has not been instantiated
                pass

    def _start_observing(self):
        self._handle = self._module.register_forward_hook(self._hook_fn)

    def _stop_observing(self):
        self._handle.remove()


class DistributionOutputFeaturesNorm(InstantaneousCallbackBasedStatistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub,
                 n_epochs: int, n_batches: int,
                 start: int, period: int,
                 name: str, module: torch.nn.Module, writer_kwargs: dict = {}):

        tag = "/".join(["Output_features_norm", name])
        super(DistributionOutputFeaturesNorm, self).__init__(platform=platform, writerstub=writerstub, tag=tag,
                                                             n_epochs=n_epochs, n_batches=n_batches,
                                                             start=start, period=period,
                                                             module=module)
        self._writer_kwargs = writer_kwargs

    def _hook_fn(self, module: torch.nn.Module, inputs: Union[Tuple[torch.Tensor, ...], torch.Tensor], outputs: Union[Tuple[torch.Tensor, ...], torch.Tensor]):

        if isinstance(outputs, Tuple):
            output = outputs[0]
        else:
            output = outputs

        if self._platform.is_multiproc_horovod_run:
            all_output = self._platform.hvd.allgather(output, name=self._tag)
        else:
            all_output = output

        norms = torch.sqrt(torch.sum(torch.pow(all_output, 2), dim=tuple(range(1, all_output.dim()))))

        # master-only point: write the statistic to disk
        if (not self._platform.is_horovod_run) or self._platform.is_master:
            try:
                self._writer.add_histogram(self._tag, norms, global_step=self._global_step, **self._writer_kwargs)
            except AttributeError:  # ``SummaryWriter`` has not been instantiated
                pass

    def _start_observing(self):
        self._handle = self._module.register_forward_hook(self._hook_fn)

    def _stop_observing(self):
        self._handle.remove()


class DistributionOutputFeaturesComponents(InstantaneousCallbackBasedStatistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub,
                 n_epochs: int, n_batches: int,
                 start: int, period: int,
                 name: str, module: torch.nn.Module, writer_kwargs: dict = {}):

        tag = "/".join(["Output_features_components", name])
        super(DistributionOutputFeaturesComponents, self).__init__(platform=platform, writerstub=writerstub, tag=tag,
                                                                   n_epochs=n_epochs, n_batches=n_batches,
                                                                   start=start, period=period,
                                                                   module=module)
        self._writer_kwargs = writer_kwargs

    def _hook_fn(self, module: torch.nn.Module, inputs: Union[Tuple[torch.Tensor, ...], torch.Tensor], outputs: Union[Tuple[torch.Tensor, ...], torch.Tensor]):

        if isinstance(outputs, tuple):
            output = outputs[0]
        else:
            output = outputs

        if self._platform.is_multiproc_horovod_run:
            all_output = self._platform.hvd.allgather(output, name=self._tag)
        else:
            all_output = output

        # master-only point: write the statistic to disk
        if (not self._platform.is_horovod_run) or self._platform.is_master:
            try:
                self._writer.add_histogram(self._tag, all_output, global_step=self._global_step, **self._writer_kwargs)
            except AttributeError:  # ``SummaryWriter`` has not been instantiated
                pass

    def _start_observing(self):
        self._handle = self._module.register_forward_hook(self._hook_fn)

    def _stop_observing(self):
        self._handle.remove()


class DistributionIncomingGradientNorm(InstantaneousCallbackBasedStatistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub,
                 n_epochs: int, n_batches: int,
                 start: int, period: int,
                 name: str, module: torch.nn.Module, writer_kwargs: dict = {}):
        """Log the histogram of the 1-D distribution of gradient components."""

        tag = "/".join(["Incoming_gradient_norm", name])
        super(DistributionIncomingGradientNorm, self).__init__(platform=platform, writerstub=writerstub, tag=tag,
                                                               n_epochs=n_epochs, n_batches=n_batches,
                                                               start=start, period=period,
                                                               module=module)
        self._writer_kwargs = writer_kwargs

    def _hook_fn(self, module: torch.nn.Module, incoming_grads: Union[Tuple[torch.Tensor, ...], torch.Tensor], outgoing_grads: Union[Tuple[torch.Tensor, ...], torch.Tensor]):

        if isinstance(incoming_grads, tuple):
            gradient = incoming_grads[0]
        else:
            gradient = incoming_grads

        if self._platform.is_multiproc_horovod_run:
            all_gradient = self._platform.hvd.allgather(gradient, name=self._tag)
        else:
            all_gradient = gradient

        norms = torch.sqrt(torch.sum(torch.pow(all_gradient, 2), dim=tuple(range(1, all_gradient.dim()))))

        # master-only point: write the statistic to disk
        if (not self._platform.is_horovod_run) or self._platform.is_master:
            try:
                self._writer.add_histogram(self._tag, norms, global_step=self._global_step, **self._writer_kwargs)
            except AttributeError:  # ``SummaryWriter`` has not been instantiated
                pass

    def _start_observing(self):
        self._handle = self._module.register_backward_hook(self._hook_fn)

    def _stop_observing(self):
        self._handle.remove()


class DistributionIncomingGradientComponents(InstantaneousCallbackBasedStatistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub,
                 n_epochs: int, n_batches: int,
                 start: int, period: int,
                 name: str, module: torch.nn.Module, writer_kwargs: dict = {}):
        """Log the histogram of the 1-D distribution of gradient components."""

        tag = "/".join(["Incoming_gradient_components", name])
        super(DistributionIncomingGradientComponents, self).__init__(platform=platform, writerstub=writerstub, tag=tag,
                                                                     n_epochs=n_epochs, n_batches=n_batches,
                                                                     start=start, period=period,
                                                                     module=module)
        self._writer_kwargs = writer_kwargs

    def _hook_fn(self, module: torch.nn.Module, incoming_grads: Union[Tuple[torch.Tensor, ...], torch.Tensor], outgoing_grads: Union[Tuple[torch.Tensor, ...], torch.Tensor]):

        if isinstance(incoming_grads, tuple):
            gradient = incoming_grads[0]
        else:
            gradient = incoming_grads

        if self._platform.is_multiproc_horovod_run:
            all_gradient = self._platform.hvd.allgather(gradient, name=self._tag)
        else:
            all_gradient = gradient

        # master-only point: write the statistic to disk
        if (not self._platform.is_horovod_run) or self._platform.is_master:
            try:
                self._writer.add_histogram(self._tag, all_gradient, global_step=self._global_step, **self._writer_kwargs)
            except AttributeError:  # ``SummaryWriter`` has not been instantiated
                pass

    def _start_observing(self):
        self._handle = self._module.register_backward_hook(self._hook_fn)

    def _stop_observing(self):
        self._handle.remove()


class DistributionOutgoingGradientNorm(InstantaneousCallbackBasedStatistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub,
                 n_epochs: int, n_batches: int,
                 start: int, period: int,
                 name: str, module: torch.nn.Module, writer_kwargs: dict = {}):
        """Log the histogram of the 1-D distribution of the gradient's norm."""

        tag = "/".join(["Outgoing_gradient_norm", name])
        super(DistributionOutgoingGradientNorm, self).__init__(platform=platform, writerstub=writerstub, tag=tag,
                                                               n_epochs=n_epochs, n_batches=n_batches,
                                                               start=start, period=period,
                                                               module=module)
        self._writer_kwargs = writer_kwargs

    def _hook_fn(self, module: torch.nn.Module, incoming_grads: Union[Tuple[torch.Tensor, ...], torch.Tensor], outgoing_grads: Union[Tuple[torch.Tensor, ...], torch.Tensor]):

        if isinstance(outgoing_grads, tuple):
            gradient = outgoing_grads[0]
        else:
            gradient = outgoing_grads

        if self._platform.is_multiproc_horovod_run:
            all_gradient = self._platform.hvd.allgather(gradient, name=self._tag)
        else:
            all_gradient = gradient

        norms = torch.sqrt(torch.sum(torch.pow(all_gradient, 2), dim=tuple(range(1, all_gradient.dim()))))

        # master-only point: write the statistic to disk
        if (not self._platform.is_horovod_run) or self._platform.is_master:
            try:
                self._writer.add_histogram(self._tag, norms, global_step=self._global_step, **self._writer_kwargs)
            except AttributeError:  # ``SummaryWriter`` has not been instantiated
                pass

    def _start_observing(self):
        self._handle = self._module.register_backward_hook(self._hook_fn)

    def _stop_observing(self):
        self._handle.remove()


class DistributionOutgoingGradientComponents(InstantaneousCallbackBasedStatistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub,
                 n_epochs: int, n_batches: int,
                 start: int, period: int,
                 name: str, module: torch.nn.Module, writer_kwargs: dict = {}):
        """Log the histogram of the 1-D distribution of gradient components."""

        tag = "/".join(["Outgoing_gradient_components", name])
        super(DistributionOutgoingGradientComponents, self).__init__(platform=platform, writerstub=writerstub, tag=tag,
                                                                     n_epochs=n_epochs, n_batches=n_batches,
                                                                     start=start, period=period,
                                                                     module=module)
        self._writer_kwargs = writer_kwargs

    def _hook_fn(self, module: torch.nn.Module, incoming_grads: Union[Tuple[torch.Tensor, ...], torch.Tensor], outgoing_grads: Union[Tuple[torch.Tensor, ...], torch.Tensor]):

        if isinstance(outgoing_grads, tuple):
            gradient = outgoing_grads[0]
        else:
            gradient = outgoing_grads

        if self._platform.is_multiproc_horovod_run:
            all_gradient = self._platform.hvd.allgather(gradient, name=self._tag)
        else:
            all_gradient = gradient

        # master-only point: write the statistic to disk
        if (not self._platform.is_horovod_run) or self._platform.is_master:
            try:
                self._writer.add_histogram(self._tag, all_gradient, global_step=self._global_step, **self._writer_kwargs)
            except AttributeError:  # ``SummaryWriter`` has not been instantiated
                pass

    def _start_observing(self):
        self._handle = self._module.register_backward_hook(self._hook_fn)

    def _stop_observing(self):
        self._handle.remove()


class MeanUpdateNormWeightNormRatio(InstantaneousCallbackFreeStatistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub,
                 n_epochs: int, n_batches: int,
                 start: int, period: int,
                 name: str, module: torch.nn.Module, writer_kwargs: dict = {}):
        """Compute the ratio between the norms of update and weight tensors.

        This statistic can help tuning the learning rate hyper-parameter.
        """

        tag = "/".join(["Update_norm_weight_norm_ratio", name])
        super(MeanUpdateNormWeightNormRatio, self).__init__(platform=platform, writerstub=writerstub, tag=tag,
                                                            n_epochs=n_epochs, n_batches=n_batches,
                                                            start=start, period=period)

        self._module = module
        assert hasattr(self._module, 'weight')

        self._writer_kwargs = writer_kwargs
        self._old_weight    = None

    def _start_observing(self):
        self._old_weight = self._module.weight.data.clone().detach().cpu()  # in multi-process data-parallel runs, each process stores a full instance of the network, whose parameters are assumed to be synchronised

    def _stop_observing(self):

        new_weight = self._module.weight.data.clone().detach().cpu()
        update     = new_weight - self._old_weight

        weight_norms = torch.sqrt(torch.sum(torch.pow(self._old_weight, 2), dim=tuple(range(1, self._old_weight.dim()))))
        update_norms = torch.sqrt(torch.sum(torch.pow(update, 2), dim=tuple(range(1, update.dim()))))
        ratios       = update_norms / weight_norms

        if (not self._platform.is_horovod_run) or self._platform.is_master:
            try:
                self._writer.add_scalar(self._tag, torch.mean(ratios), global_step=self._global_step, **self._writer_kwargs)
            except AttributeError:  # ``SummaryWriter`` has not been instantiated
                pass

