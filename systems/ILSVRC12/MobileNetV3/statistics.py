from typing import Union, Tuple
import torch
from torch import nn
from manager.meter import WriterStub
from manager.meter.statistics import InstantaneousCallbackBasedStatistic, InstantaneousCallbackFreeStatistic
from manager.platform import PlatformManager
from quantlib.editing.lightweight import LightweightGraph
from quantlib.editing.lightweight.rules import SubTypeFilter
from quantlib.algorithms.bb.bb_ops import *
from quantlib.algorithms.pact.pact_ops import *

class BBGateStatistic(InstantaneousCallbackBasedStatistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub,
                 n_epochs: int, n_batches: int,
                 start: int, period: int, kind : str,
                 name: str, module: torch.nn.Module, writer_kwargs: dict = {}):

        tag = f"{kind}/{name}/bb_gates"
        super(BBGateStatistic, self).__init__(platform=platform, writerstub=writerstub, tag=tag,
                                                             n_epochs=n_epochs, n_batches=n_batches,
                                                             start=start, period=period,
                                                             module=module)
        self._writer_kwargs = writer_kwargs

    def _fw_hook_fn(self, module: torch.nn.Module, inputs: Union[Tuple[torch.Tensor, ...], torch.Tensor], outputs: Union[Tuple[torch.Tensor, ...], torch.Tensor]):

        for g, p in zip(module.bb_gates, module.precs[1:]):
            if (not self._platform.is_horovod_run) or self._platform.is_master:
                try:
                    self._writer.add_scalar(self._tag+f"/gate_{p}b", g.item(), global_step=self._global_step, **self._writer_kwargs)
                except AttributeError:  # ``SummaryWriter`` has not been instantiated
                    print("OI NO WRITER")

    def _bw_hook_fn(self, grad):
        for g, p in zip(grad, self._module.precs[1:]):
            if (not self._platform.is_horovod_run) or self._platform.is_master:
                try:
                    wr = self._writer
                except AttributeError:  # ``SummaryWriter`` has not been instantiated
                    print("OI NO WRITER")
                    wr = None
                if wr:
                    wr.add_scalar("grads/"+self._tag+f"/gate_{p}b", g.item(), global_step=self._global_step, **self._writer_kwargs)



    def _start_observing(self):
        self._fw_handle = self._module.register_forward_hook(self._fw_hook_fn)
        self._bw_handle = self._module.bb_gates.register_hook(self._bw_hook_fn)


    def _stop_observing(self):
        self._fw_handle.remove()
        self._bw_handle.remove()

class BBGateMasterStatistic(InstantaneousCallbackFreeStatistic):
    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub,
                 n_epochs: int, n_batches: int,
                 start: int, period: int,
                 name: str, module: torch.nn.Module, writer_kwargs: dict = {}):
        tag = f"{name}/all_bb_gates"
        super(BBGateMasterStatistic, self).__init__(platform=platform, writerstub=writerstub, tag=tag,
                                                             n_epochs=n_epochs, n_batches=n_batches,
                                                             start=start, period=period
                                                             )
        self._writer_kwargs = writer_kwargs
        nodes = LightweightGraph.build_nodes_list(module)
        act_filter = SubTypeFilter(BBAct)
        conv_filter = SubTypeFilter(BBConv2d)
        lin_filter = SubTypeFilter(BBLinear)
        self.act_keys = [n.name for n in act_filter(nodes)]
        self.conv_keys = [n.name for n in conv_filter(nodes)]
        self.lin_keys = [n.name for n in lin_filter(nodes)]
        self.module = module
        self.prefix = name+'.' if name is not None else ''

    def _start_observing(self):
        pass


    def write(self, module : nn.Module, kind : str, name : str):
        for g, p in zip(module.bb_gates, module.precs[1:]):
            if (not self._platform.is_horovod_run) or self._platform.is_master:
                try:
                    self._writer.add_scalar(f"{kind}/{self.prefix}{name}/gate_{p}b", g.item(), global_step=self._global_step, **self._writer_kwargs)
                except AttributeError:  # ``SummaryWriter`` has not been instantiated
                      print("OI NO WRITER")

    def _stop_observing(self):
        for kind, keys in [('act', self.act_keys), ('conv', self.conv_keys), ('linear', self.lin_keys)]:
            for k in keys:
                m = self.module.get_submodule(k)
                self.write(m, kind, k)


class TQTMasterStatistic(InstantaneousCallbackFreeStatistic):
    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub,
                 n_epochs: int, n_batches: int,
                 start: int, period: int,
                 name: str, module: torch.nn.Module, writer_kwargs: dict = {}):
        tag = f"{name}/all_tqt"

        super(TQTMasterStatistic, self).__init__(platform=platform, writerstub=writerstub, tag=tag,
                                                             n_epochs=n_epochs, n_batches=n_batches,
                                                             start=start, period=period
                                                             )
        self._writer_kwargs = writer_kwargs
        nodes = LightweightGraph.build_nodes_list(module)
        act_filter = SubTypeFilter(_PACTActivation)
        conv_filter = SubTypeFilter(PACTConv2d)
        lin_filter = SubTypeFilter(PACTLinear)
        self.act_keys = [n.name for n in act_filter(nodes)]
        self.conv_keys = [n.name for n in conv_filter(nodes)]
        self.lin_keys = [n.name for n in lin_filter(nodes)]
        self.module = module
        self.prefix = name+'.' if name is not None else ''


    def write(self, module : nn.Module, kind : str, name : str):
        if (not self._platform.is_horovod_run) or self._platform.is_master:
            try:
                self._writer.add_scalar(f"clip_{kind}/{self.prefix}{name}/clip_hi", torch.max(module.clip_hi), global_step=self._global_step, **self._writer_kwargs)
                self._writer.add_scalar(f"clip_{kind}/{self.prefix}{name}/clip_lo", torch.min(module.clip_lo), global_step=self._global_step, **self._writer_kwargs)
                if module.tqt:
                    self._writer.add_scalar(f"clip_{kind}/{self.prefix}{name}/log_t", torch.max(module.log_t), global_step=self._global_step, **self._writer_kwargs)
            except AttributeError:  # ``SummaryWriter`` has not been instantiated
                print("OI NO WRITER")

    def _start_observing(self):
        pass
    def _stop_observing(self):
        for kind, keys in [('act', self.act_keys), ('conv', self.conv_keys), ('linear', self.lin_keys)]:
            for k in keys:
                m = self.module.get_submodule(k)
                self.write(m, kind, k)
