import torch

from systems.GoogleSpeech.utils.statistics import GoogleSpeechStatistic

from manager.platform import PlatformManager
from manager.meter import WriterStub


def _postprocess_gt(ygt: torch.Tensor) -> torch.Tensor:
    return ygt.unsqueeze(-1)


def _postprocess_pr(ypr: torch.Tensor) -> torch.Tensor:
    return ypr.argmax(dim=1).unsqueeze(-1)


class DSCNNStatistic(GoogleSpeechStatistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub,
                 n_epochs: int, n_batches: int,
                 train: bool):
        super(DSCNNStatistic, self).__init__(platform=platform, writerstub=writerstub,
                                             n_epochs=n_epochs, n_batches=n_batches,
                                             train=train,
                                             postprocess_gt_fun=_postprocess_gt, postprocess_pr_fun=_postprocess_pr)
