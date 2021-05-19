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
