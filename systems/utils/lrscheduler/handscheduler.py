import torch
from torch.optim.lr_scheduler import _LRScheduler

from typing import List


class HandScheduler(_LRScheduler):

    def __init__(self, optimizer: torch.optim.Optimizer, schedule: dict, last_epoch: int = -1):
        """Scale the learning rates by manually-set factors.

        Remember: the schedule specification :math:`(e, s_{e})` amounts to
        instructing "at the END of the epoch with identifier :math:`e`,
        multiply all the base learning rates in the different optimizer
        parameter groups by the factor :math:`s_{e}`".
        """

        self.last_epoch = last_epoch
        self._schedule  = {int(k): v for k, v in schedule.items()}  # parse string keys to ints

        super(HandScheduler, self).__init__(optimizer, last_epoch=last_epoch)

    def get_lrs(self, just_finished_epoch: int) -> List[float]:
        return list(map(lambda base_lr: base_lr * self._schedule[just_finished_epoch], self.base_lrs))

    def step(self, epoch: int = None):

        if epoch is not None:
            self.last_epoch = epoch

        just_finished_epoch = self.last_epoch + 1

        if just_finished_epoch in self._schedule.keys():
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lrs(just_finished_epoch)):
                param_group['lr'] = lr

        self.last_epoch = just_finished_epoch
