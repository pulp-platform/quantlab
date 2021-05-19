import torch

from systems.CIFAR10.utils.transforms.transforms import CIFAR10STATS
from manager.meter import RGBInputsSnapshot

from manager.platform import PlatformManager
from manager.meter import WriterStub


class CIFAR10InputsSnapshot(RGBInputsSnapshot):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub,
                 n_epochs: int, n_batches: int,
                 start: int, period: int,
                 name: str, module: torch.nn.Module, n_inputs: int, preprocessing_type: int, writer_kwargs: dict = {}):

        super(CIFAR10InputsSnapshot, self).__init__(platform=platform, writerstub=writerstub,
                                                    n_epochs=n_epochs, n_batches=n_batches,
                                                    start=start, period=period,
                                                    name=name, module=module, n_inputs=n_inputs, preprocessing_type=preprocessing_type, preprocessing_stats=CIFAR10STATS['normalize'], writer_kwargs=writer_kwargs)
