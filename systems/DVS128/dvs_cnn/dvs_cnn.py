import torch
import torch.nn as nn
import numpy as np
from quantlib.algorithms.generic import CausalConv1d
from quantlib.QTensor import QTensor
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


import os
from pathlib import Path

__CNN_CFGS__ = {
    'first_try' : [128, 128, 128, 128],
    'ninetysix_ch' : [96, 96, 96, 96],
    'eighty_ch' : [80, 80, 80, 80],
    'reduced_channels' : [64, 64, 64, 64],
    '128_channels' : [128, 128, 128, 128],
    '96_channels' : [96, 96, 96, 96],
    '64_channels' : [64, 64, 64, 64],
    '48_channels' : [48, 48, 48, 48],
    '32_channels' : [32, 32, 32, 32]
}

# TCN configs:
# (kernel_size, dilation, output_channels)
__TCN_CFGS__ = {
    'first_try' : [(2, 1, 64), (2, 2, 64), (2, 4, 64)],
    '64_channels' : [(2, 1, 64), (2, 2, 64), (2, 4, 64)],
    '64_channels_k3' : [(3, 1, 64), (3, 2, 64), (3, 4, 64)],
    '64_channels_k4' : [(4, 1, 64), (4, 2, 64), (4, 4, 64)],
    'eighty_ch' : [(2, 1, 80), (2, 2, 80), (2, 4, 80)],
    '128_ch' : [(2, 1, 128), (2, 2, 128), (2, 4, 128)],
    '128_channels' : [(2, 1, 128), (2, 2, 128), (2, 4, 128)],
    'k3' : [(3, 1, 64), (3, 2, 64), (3, 4, 64)],
    '96_channels' : [(2, 1, 96), (2, 2, 96), (2, 4, 96)],
    '96_channels_k3' : [(3, 1, 96), (3, 2, 96), (3, 4, 96)],
    '80_channels_k3' : [(3, 1, 80), (3, 2, 80), (3, 4, 80)],
    '96_channels_k4' : [(4, 1, 96), (4, 2, 96), (4, 4, 96)],
    '32_channels' : [(2, 1, 32), (2, 2, 32), (2, 4, 32)],
    '32_channels_k3' : [(3, 1, 32), (3, 2, 32), (3, 4, 32)],
    '32_channels_k4' : [(4, 1, 32), (4, 2, 32), (4, 4, 32)],
    '48_channels' : [(2, 1, 48), (2, 2, 48), (2, 4, 48)],
    '48_channels_k3' : [(3, 1, 48), (3, 2, 48), (3, 4, 48)],
    '48_channels_k4' : [(4, 1, 48), (4, 2, 48), (4, 4, 48)]
}

def get_input_shape(cfg : dict):
    cnn_win = cfg['network']['kwargs']['cnn_window']
    tcn_win = cfg['network']['kwargs']['tcn_window']
    ds = cfg['data']['train']['dataset']['transform']['kwargs']['downsample']
    xy_in = 128//ds
    return (1, cnn_win*tcn_win, xy_in, xy_in)

class DVSNet2D(nn.Module):
    def __init__(self, cnn_cfg_key : str, pool_type : str = "stride", cnn_window : int = 16, activation : str = 'relu',
                 out_size : int = 11, use_classifier : bool = True, fix_cnn_pool=False, k : int = 3, layer_order : str = 'pool_bn', last_conv_nopad : bool = False, adapter_out_ch : int = 32, **kwargs):
        super(DVSNet2D, self).__init__()
        cfg = __CNN_CFGS__[cnn_cfg_key]
        self.k = k
        if activation == 'relu':
            self._act = nn.ReLU
        elif activation == 'relu6':
            self._act = nn.ReLU6
        elif activation == 'htanh':
            self._act = nn.Hardtanh
        else:
            assert False, "Invalid activation function supplied: {}".format(activation)

        
        assert layer_order in ['pool_bn', 'bn_pool'], "Invalid layer order specified: {}".format(layer_order)

        adapter_list = []

        adapter_list.append(nn.Conv2d(cnn_window, adapter_out_ch, kernel_size=k, padding=k//2, bias=False))
        if pool_type != 'max_pool':
            adapter_pool = nn.AvgPool2d(kernel_size=2)
        else:
            adapter_pool = nn.MaxPool2d(kernel_size=2)
        if layer_order == 'pool_bn':
            adapter_list.append(adapter_pool)
            adapter_list.append(nn.BatchNorm2d(adapter_out_ch))
        else:
            adapter_list.append(nn.BatchNorm2d(adapter_out_ch))
            adapter_list.append(adapter_pool)
        adapter_list.append(self._act(inplace=True))
        adapter = nn.Sequential(*adapter_list)
        self.adapter = adapter

        features = self._get_features(adapter_out_ch, cfg, k, pool_type, self._act, layer_order, last_conv_nopad)
        self.features = features
        # after features block, we should have a 4x4 feature map
        if use_classifier:
            self.classifier = nn.Linear(cfg[-1]*16, out_size)
            self.out_pool = None
        else:
            self.classifier = None
            #if not using a classifier, we want to output a (N_B, cfg[-1], 1, 1)
            #vector, so average pool everything together
            # OI M8 U CAN'T DO THAT!!!
            # every layer needs to be a stack of conv(->pool)->act to produce
            #ternary activations
            if not last_conv_nopad:
                if not fix_cnn_pool:
                    self.out_pool = nn.AdaptiveAvgPool2d((1,1))
                else:
                    # make the "out_pool" layer a conforming stack
                    out_pool_conv = nn.Conv2d(cfg[-1], cfg[-1], kernel_size=(1,1), stride=(1,1), padding=0, dilation=1, bias=False)
                    out_pool_layer = nn.AdaptiveAvgPool2d((1,1))
                    out_pool_bn = nn.BatchNorm2d(cfg[-1])
                    out_pool_act = self._act(inplace=True)
                    self.out_pool = nn.Sequential(out_pool_conv, out_pool_layer, out_pool_bn, out_pool_act)
            else:
                # last_conv_nopad requires the input size to be (64, 64),
                # otherwise the output dimensions don't match, so we add this
                # pooling layer to make it work even for different input sizes
                self.out_pool = nn.AdaptiveAvgPool2d((1,1))


    @staticmethod
    def _get_features(in_channels : int, cfg : list, k : int, pool_type : str = "stride", act : type = nn.ReLU, layer_order : str = 'pool_bn', last_conv_nopad : bool = False):
        l = []
        for i, c in enumerate(cfg):
            if not (i == len(cfg)-1 and last_conv_nopad):
                pad = k//2
            else:
                pad = 0
            if pool_type == "stride":
                l.append(nn.Conv2d(in_channels, c, kernel_size=k, stride=2, padding=pad, bias=False))
            else:
                if pool_type == "avg_pool":
                    pool_l = nn.AvgPool2d(kernel_size=2)
                else:
                    pool_l = nn.MaxPool2d(kernel_size=2)
                l.append(nn.Conv2d(in_channels, c, kernel_size=k, stride=1, padding=pad, bias=False))
                if layer_order == "pool_bn":
                    l.append(pool_l)
            l.append(nn.BatchNorm2d(c))
            if layer_order == "bn_pool" and pool_type != "stride":
                l.append(pool_l)
            l.append(act(inplace=True))
            in_channels = c
        return nn.Sequential(*l)

    def forward(self, x):
        x = self.adapter(x)
        x = self.features(x)
        if self.classifier is not None:
            x = torch.flatten(x, start_dim=1)
            x = self.classifier(x)
            return x
        else:
            if self.out_pool is not None:
                x = self.out_pool(x)
            # return a 2D tensor
            # shape: (n_batch, cfg[-1])
            x = torch.flatten(x, start_dim=1)
            return x


class DVSTCN(nn.Module):
    def __init__(self, tcn_cfg_key : str, in_channels : int, activation : str = 'relu', n_classes : int = 11, sequence_length : int = 5, classifier_type : str = "learned", classifier_out : str = "all", classifier_bias : bool = True, twn_classifier : bool = False, **kwargs):
        # classifier_type can be "learned" or "linear"
        # classifier_out can be "all" (output as many classifications as
        # sequence elements) or "last" (output only the classification vector
        # of the last sequence element). If classifier_type is "linear",
        # classifier_out has no effect (as "linear" only produces one
        # classification anyway)
        super(DVSTCN, self).__init__()
        cfg = __TCN_CFGS__[tcn_cfg_key]
        if activation == 'relu':
            self._act = nn.ReLU
        elif activation == 'relu6':
            self._act = nn.ReLU6
        elif activation == 'htanh':
            self._act = nn.Hardtanh
        else:
            assert False, "Invalid activation function supplied: {}".format(activation)

        self.n_classes = n_classes
        self.twn_classifier = twn_classifier
        self.sequence_length = sequence_length
        assert classifier_out in ["all", "last"], "DVSTCN parameter classifier_out must be 'all' or 'last'!"
        self.classifier_type = classifier_type
        self.classifier_out = classifier_out
        features = self._get_features(in_channels, cfg, self._act, twn_classifier)
        self.features = features
        classifier = self._get_classifier(classifier_type, cfg[-1][-1], n_classes, classifier_bias, sequence_length)
        self.classifier = classifier
        # this must set to true if "linear" (conv1d module) classifier is replaced
        # with a Linear module
        self.cls_replaced = False

    @staticmethod
    def _get_classifier(classifier_type, in_ch, n_classes, classifier_bias, sequence_length):
        # classifier_type can be:
        #  - learned  -> each sequence element is mapped to a one-hot classification output with a learned K=1 conv1d
        #  - linear   -> all input sequence elements are fed together into a linear layer to produce a single output classification
        if classifier_type == "linear":
            k = sequence_length
        else:
            k = 1
        classifier = nn.Conv1d(in_ch, n_classes, k, padding=0, dilation=1, bias=classifier_bias)

        return classifier

    @staticmethod
    def _get_features(in_channels : int, cfg : list, act : type = nn.ReLU, twn_classifier : bool = False):
        l = []
        ic = in_channels
        for i, c in enumerate(cfg):
            k, dil, oc = c
            # for causal convolution, pad on the left only
            lpad = (k-1)*dil
            l.append(CausalConv1d(ic, oc, k, stride=1, dilation=dil, bias=False))
            # if we want to use the TWN classifier, don't put batchnorm after
            # the last feature layer
            if i != len(cfg)-1 or not twn_classifier:
                l.append(nn.BatchNorm1d(oc))
                l.append(act())
            ic = oc
        return nn.Sequential(*l)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        if self.cls_replaced:
            return x
        if self.classifier_type == "linear":
            return x.squeeze(2)
        elif self.classifier_out == "last":
            return x[:, :, -1]
        return x


class DVSHybridNet(nn.Module):
    def __init__(self, cnn_window : int, tcn_window : int, cnn_cfg_key : str, tcn_cfg_key : str, n_classes : int = 11, activation : str = 'relu', k_cnn : int = 3, last_conv_nopad : bool = False, pretrained : str = None, inject_eps : bool = False, use_tcn : bool = True, classifier_bias : bool = False, fold_id : int = 0, **kwargs):
        super(DVSHybridNet, self).__init__()
        cnn_cfg = __CNN_CFGS__[cnn_cfg_key]
        embedding_size = cnn_cfg[-1]
        self.cnn_window = cnn_window
        self.tcn_window = tcn_window
        self.sequence_length = tcn_window
        self.use_tcn = use_tcn
        self.cnn = DVSNet2D(cnn_cfg_key=cnn_cfg_key, cnn_window=cnn_window, use_classifier=False, activation=activation, k=k_cnn, last_conv_nopad=last_conv_nopad, **kwargs)
        if use_tcn:
            self.tcn = DVSTCN(tcn_cfg_key=tcn_cfg_key, in_channels=embedding_size, n_classes=n_classes, activation=activation, sequence_length=tcn_window, classifier_bias=classifier_bias, **kwargs)
        else:
            # if we don't use a true TCN, just use a single Conv1D layer
            # (equivalent to a linear layer)
            self.tcn = nn.Conv1d(cnn_cfg[-1], n_classes, tcn_window, padding=0, dilation=1, bias=classifier_bias)
        self.inject_eps = inject_eps

        def get_max_acc_ckpt(exp_dir : Path):
            log_dir = exp_dir.joinpath(f"fold{fold_id}/stats/epoch")
            saves_dir = exp_dir.joinpath(f"fold{fold_id}/saves")
            ea = EventAccumulator(str(log_dir)).Reload()
            max_step, max_acc = max(((evt.step, evt.value) for evt in ea.Scalars("Accuracy/Valid") if saves_dir.joinpath(f'epoch{evt.step:03}.ckpt').exists()), key=lambda tpl:tpl[1])
            load_opts = {}
            if not torch.cuda.is_available():
                load_opts['map_location'] = torch.device('cpu')
            best_acc_state_dict = torch.load(saves_dir.joinpath(f'epoch{max_step:03}.ckpt'), **load_opts)['net']
            # convert keys for nn.DataParallel
            if all(k.startswith('module.') for k in best_acc_state_dict.keys()):
                best_acc_state_dict = {k[7:] : v for k,v in best_acc_state_dict.items()}
            return best_acc_state_dict
        if pretrained:
            pretrained = Path(pretrained)
            # if pretrained is not a file but a directory, assume it is a QL
            # log directory and get the pretrained checkpoint for the specified
            # fold
            if pretrained.is_dir():
                ckpt = get_max_acc_ckpt(pretrained)
            elif pretrained.is_file():
                load_opts = {}
                if not torch.cuda.is_available():
                    load_opts['map_location'] = torch.device('cpu')
                ckpt = torch.load(pretrained, **load_opts)
            self.load_state_dict(ckpt)


    def forward(self, x):
        # we get a (cnn_window * tcn_window)-sized stack of frame batches
        # => shape: (n_batch, cnn_window*tcn_window, H, W)
        # 1. split it up into cnn_window-sized stacks

        if self.inject_eps:
            x = QTensor(x, eps=1.)
        #print(f"type of x - hybridnet: {type(x)}")
        cnn_wins = torch.split(x, self.cnn_window, dim=1)
        #print(f"eps of x - hybridnet after split: {tuple(w.eps for w in cnn_wins)}")

        # 2. run the CNN over each of the chunks
        cnn_outs = []
        for w in cnn_wins:
            cnn_outs.append(self.cnn(w))
        # 3. assemble the CNN outputs into a "window":
        # => shape: (n_batch, tcn_window, cnn_cfg[-1])
        cnn_output = torch.stack(cnn_outs, dim=2)

        # 4. run the TCN over this window
        tcn_output = self.tcn(cnn_output)
        if not self.use_tcn:
            tcn_output = tcn_output.squeeze(2)
        # 5. done! We get an output window:
        #    shape: (n_batch, n_classes, w)
        #           where w == tcn_window if tcn.classifier_out == "all" and w == 1 if tcn.classifier_out == "last"
        return tcn_output

