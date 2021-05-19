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
