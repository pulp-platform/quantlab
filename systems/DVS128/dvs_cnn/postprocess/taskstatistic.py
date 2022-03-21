from manager.platform import PlatformManager
from manager.meter import WriterStub

from systems.DVS128.utils.statistics import DVS128Statistic

def _postprocess_pr(pr_outs):
    idxs = pr_outs.detach().cpu().max(1).indices
    return idxs


def _postprocess_gt(gt_labels):
    return gt_labels.detach().cpu()


class dvs_cnnStatistic(DVS128Statistic):

    def __init__(self,
                 platform: PlatformManager, writerstub: WriterStub,
                 n_epochs: int, n_batches: int,
                 train: bool):
        super(dvs_cnnStatistic, self).__init__(platform=platform, writerstub=writerstub,
                                           n_epochs=n_epochs, n_batches=n_batches,
                                           train=train,
                                           postprocess_gt_fun=_postprocess_gt, postprocess_pr_fun=_postprocess_pr)

