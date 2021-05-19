import torch
import torch.nn as nn
from functools import partial

from .rules import LightweightRule
import quantlib.algorithms as qa

from .filters import Filter
from typing import Union


def replace_conv2d_inqconv2d(module: torch.nn.Module,
                             num_levels: int,
                             quant_init_method: Union[str, None],
                             quant_strategy: str) -> torch.nn.Module:

    assert type(module) == nn.Conv2d

    return qa.inq.INQConv2d(in_channels=module.in_channels,
                            out_channels=module.out_channels,
                            kernel_size=module.kernel_size,
                            stride=module.stride,
                            padding=module.padding,
                            dilation=module.dilation,
                            groups=module.groups,
                            bias=module.bias,
                            num_levels=num_levels,
                            quant_init_method=quant_init_method,
                            quant_strategy=quant_strategy)


class INQConv2dRule(LightweightRule):

    def __init__(self,
                 filter_: Filter,
                 num_levels: int,
                 quant_init_method: Union[str, None] = None,
                 quant_strategy: str = 'magnitude'):

        replacement_fun = partial(replace_conv2d_inqconv2d, num_levels=num_levels, quant_init_method=quant_init_method, quant_strategy=quant_strategy)  # currying returns a function with the correct signature ``Callable[[torch.nn.Module], torch.nn.Module]``
        super(INQConv2dRule, self).__init__(filter_=filter_,
                                            replacement_fun=replacement_fun)
