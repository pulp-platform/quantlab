from typing import Optional, Dict

import torch
from torch import nn


_ACTIVATIONS = {
    "hswish" : nn.Hardswish,
    "hsigm" : nn.Hardsigmoid,
    "relu" : nn.ReLU
}

_CONFIGS = {
    "small" : {
        "blocks" : [
            [3,  16,    1, "relu",  2, True ],
            [3,  24,  4.5, "relu",  2, False],
            [3,  24, 3.67, "relu",  1, False],
            [5,  40,    4, "hswish", 2, True ],
            [5,  40,    6, "hswish", 1, True ],
            [5,  40,    6, "hswish", 1, True ],
            [5,  48,    3, "hswish", 1, True ],
            [5,  48,    3, "hswish", 1, True ],
            [5,  96,    6, "hswish", 2, True ],
            [5,  96,    6, "hswish", 1, True ],
            [5,  96,    6, "hswish", 1, True ],
        ],
        "classifier_dim" : 1024
    },
    "large" : {
        "blocks": [
            [3,  16,   1, "relu",  1,  False],
            [3,  24,   4, "relu",  2,  False],
            [3,  24,   3, "relu",  1,  False],
            [5,  40,   3, "relu",  2,  True ],
            [5,  40,   3, "relu",  1,  True ],
            [5,  40,   3, "relu",  1,  True ],
            [3,  80,   6, "hswish", 2,  False],
            [3,  80, 2.5, "hswish", 1,  False],
            [3,  80, 2.3, "hswish", 1,  False],
            [3,  80, 2.3, "hswish", 1,  False],
            [3, 112,   6, "hswish", 1,  True ],
            [3, 112,   6, "hswish", 1,  True ],
            [5, 160,   6, "hswish", 2,  True ],
            [5, 160,   6, "hswish", 1,  True ],
            [5, 160,   6, "hswish", 1,  True ]
        ],
        "classifier_dim" : 1280
    }
}

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcite(nn.Module):
    def __init__(self, channels : int, reduction : int = 4):
        super(SqueezeExcite, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.lin = nn.Sequential(
            nn.Conv2d(channels, _make_divisible(channels//reduction, 8), 1),
            nn.ReLU(),
            nn.Conv2d(_make_divisible(channels//reduction, 8), channels, 1),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        pooled = self.avgpool(x)
        exc = self.lin(pooled)
        return x * exc

class ConvBnAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch : int, k: int, s : int, act : str, groups : int = 1, bn_cfg : Optional[Dict] = None):
        if bn_cfg is None:
            bn_cfg = {"eps" : 0.001, "momentum":0.01}
        nonlin = _ACTIVATIONS[act] if act is not None else None

        layers = []
        layers.append(nn.Conv2d(in_ch, out_ch, k, s, (k-1)//2, groups=groups, bias=False))
        layers.append(nn.BatchNorm2d(out_ch, **bn_cfg))
        if nonlin is not None:
            layers.append(nonlin())
        super(ConvBnAct, self).__init__(*layers)

class InvertedResidual(nn.Module):
    def __init__(self, k : int, in_ch : int, int_ch : int, out_ch : int, act : str, s : int, use_se : bool):
        super(InvertedResidual, self).__init__()
        nonlin = _ACTIVATIONS[act]
        # use residual connection only if feature map size and # input and
        # output channels are the same
        self.residual = (in_ch == out_ch) and (s == 1)
        layers = []
        if in_ch != int_ch:
            # pointwise/expansion
            layers.append(ConvBnAct(in_ch, int_ch, 1, 1, act))
        # depthwise -- nonlinearity before squeeze-excite as is done in TorchVision version
        layers.append(ConvBnAct(int_ch, int_ch, k, s, act, int_ch))
        if use_se:
            layers.append(SqueezeExcite(int_ch))
        # pointwise/projection -- no output activation!
        layers.append(ConvBnAct(int_ch, out_ch, 1, 1, None))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        y = self.block(x)
        if self.residual:
            return y+x
        else:
            return y

class MobileNetV3(nn.Module):
    def __init__(self, config : str, in_ch : int = 3, first_conv_ch : int = 16, n_classes : int = 1000, seed : int = -1, width_mult : float = 1., pretrained : Optional[str] = None):
        super(MobileNetV3, self).__init__()
        first_conv_ch = _make_divisible(first_conv_ch * width_mult, 8)
        self.pilot = ConvBnAct(in_ch, first_conv_ch, 3, 2, "hswish")

        blk_in_ch = first_conv_ch
        cfg = _CONFIGS[config]
        block_cfgs = cfg["blocks"]
        feature_layers = []
        for k, out_ch, exp, act, s, use_se in block_cfgs:
            out_ch = _make_divisible(width_mult * out_ch, 8)
            int_ch = _make_divisible(width_mult * exp * blk_in_ch, 8)
            feature_layers.append(InvertedResidual(k, blk_in_ch, int_ch, out_ch, act, s, use_se))
            blk_in_ch = out_ch

        feature_layers.append(ConvBnAct(out_ch, int_ch, 1, 1, "hswish"))
        self.features = nn.Sequential(*feature_layers)

        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                     nn.Flatten())

        cls_dim = cfg["classifier_dim"]
        self.classifier = nn.Sequential(
            nn.Linear(int_ch, cls_dim),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(cls_dim, n_classes))

        if pretrained:
            self.load_state_dict(torch.load(pretrained))
        else:
            self._initialize_weights(seed)

    def _initialize_weights(self, seed):
        if seed >= 0:
            torch.manual_seed(seed)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pilot(x)
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)


