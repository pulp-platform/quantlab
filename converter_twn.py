import os
import sys

import math
import json
import importlib
import torch
import numpy as np

from backends.twn_accelerator.compiler_vgg import compile_vgg
import shutil

sys.path.insert(0, os.pardir)  # make QuantLab packages accessible


__MASTER_PROC_RANK__ = 0
__MAX_EXPERIMENTS__  = 1000
__ALIGN_EXP__        = math.ceil(math.log10(__MAX_EXPERIMENTS__))  # experiment ID string length (decimal literal)
__MAX_CV_FOLDS__     = 10
__ALIGN_CV_FOLDS__   = math.ceil(math.log10(__MAX_CV_FOLDS__))  # cross-validation fold ID string length (decimal literal)


with open('backends/twn_accelerator/source.json') as fp:
    specs = json.load(fp)


class MiniLogbook(object):
    def __init__(self, specs):
        self.lib = importlib.import_module('.'.join(['problems', specs['problem'], specs['topology']]))
        self.dir_data = os.path.join(os.path.dirname(os.path.dirname(self.lib.__file__)), 'data')
        self.dir_exp = os.path.join(os.path.dirname(os.path.dirname(self.lib.__file__)),
                                                    'logs',
                                                    'exp{}'.format(str(specs['exp_id']).rjust(__ALIGN_EXP__, '0')))
        self.config = None
        with open(os.path.join(self.dir_exp, 'config.json')) as fp:
            self.config = json.load(fp)

        ckpt_file = os.path.join(self.dir_exp,
                                 'fold{}'.format(str(specs['fold_id']).rjust(__ALIGN_CV_FOLDS__, '0')),
                                 'saves',
                                 specs['ckpt'])
        self.ckpt = torch.load(ckpt_file)


logbook = MiniLogbook(specs)


# create the network and quantize (if specified), then load trained parameters
net = getattr(logbook.lib, logbook.config['network']['class'])(**logbook.config['network']['params'])
if logbook.config['network']['quantize'] is not None:
    quant_convert = getattr(logbook.lib, logbook.config['network']['quantize']['routine'])
    net = quant_convert(logbook.config['network']['quantize'], net)

net.load_state_dict(logbook.ckpt['network'])
net.eval()  # freeze batch-norm parameters
for n in net.named_modules():
    if hasattr(n[1], 'started'):  # put STE nodes in "quantized mode"
        n[1].started = True

train_set, valid_set = logbook.lib.load_data_sets(logbook)


# compile VGG
def apply_ste_postproc(x, ste_n, ste_m):
    ex = (2 * ste_m) / (ste_n - 1)
    return (x * ex).to(torch.float32)


def revert_ste_postproc(x, ste_n, ste_m):
    ex = (2 * ste_m) / (ste_n - 1)
    return (x / ex).to(torch.float64).round()


def convert_input_image(img, input_type):
    """Rescale a normalised ImageNet data point to UINT8 or INT8 range."""
    if not ('int' in input_type):
        return img
    else:
        from problems.ImageNet.VGG.preprocess import _ImageNet
        mean = np.array(_ImageNet['Normalize']['mean']) * 255.
        std = np.array(_ImageNet['Normalize']['std']) * 255.

        new_img = img.squeeze(0)
        new_img = (new_img.permute(1, 2, 0) * std + mean).permute(2, 0, 1).clamp(min=0., max=255.).round()
        new_img = new_img.unsqueeze(0)

        if input_type == 'int8':  # signed integer
            new_img -= 128.

    return new_img


input_type = 'uint8'
output_dir = 'trialVGG'
tq_net, fq_net = compile_vgg(net, output_dir=output_dir, input_type=input_type)

n_trials = 1
match = 0
for i in range(n_trials):
    # i = torch.randint(low=0, high=len(valid_set), size=(1,))
    img = valid_set[i][0].unsqueeze(0)

    fq_x_in = img.clone().to(torch.float32)
    fq_x = fq_x_in.clone()

    tq_x_in = convert_input_image(fq_x_in.clone(), input_type)
    tq_x_in = tq_x_in.to(torch.float64)
    tq_x = tq_x_in.clone()

    errors = []
    for l, (tql, fql) in enumerate(zip(tq_net, fq_net)):

        tql = tql.to(torch.float64)
        fql = fql.to(torch.float32)

        tq_x = tql(tq_x)

        fq_x = fql(fq_x)

        if l < 15:
            ste = fql[-1]
            fq_x_ck = revert_ste_postproc(fq_x, ste.num_levels, ste.abs_max_value)

            diff = (tq_x - fq_x_ck).detach().numpy()
            diff_stats = np.histogram(diff, np.arange(-ste.num_levels, ste.num_levels))
            errors.append((diff, diff_stats))
            print("Layer {:0>2} - Percentage error: {:6.4f}%".format(l, 100 * (np.count_nonzero(diff) / diff.size)))

        if l == 15:
            bs = tq_x.shape[0]
            tq_x = tq_x.view(bs, 1, -1)
            fq_x = fq_x.view(bs, 1, -1)

    fq_result = torch.argmax(fq_x)
    tq_result = torch.argmax(tq_x)

    print("Image {} - TQNet: {}, FQNet: {}".format(i, tq_result.item(), fq_result.item()))
    match += int(tq_result.item()) == int(fq_result.item())

# shutil.rmtree(output_dir)


from backends.twn_accelerator.debug import get_operands_fq, get_operands_tq

tq_out1 = tq_net[0](tq_x_in)

fq_out1 = fq_net[0](fq_x_in)
n_out1 = fq_net[0][-1].num_levels
m_out1 = fq_net[0][-1].abs_max_value
fq_out1_ck = revert_ste_postproc(fq_out1, n_out1, m_out1)

diff1 = (tq_out1 - fq_out1_ck).detach().numpy()
maxdiff1 = np.max(np.abs(diff1))
coords1 = list(zip(*np.where(np.abs(diff1) == maxdiff1)))

tq_1_ops = get_operands_tq(coords1[0], tq_x_in, tq_net[0], d2d_layer=False)
fq_1_ops = get_operands_fq(coords1[0], fq_x_in, fq_net[0], inq_layer=False)

tq_out2 = tq_net[1](tq_out1)

fq_out2 = fq_net[1](fq_out1)
n_out2 = fq_net[1][-1].num_levels
m_out2 = fq_net[1][-1].abs_max_value
fq_out2_ck = revert_ste_postproc(fq_out2, n_out2, m_out2)

diff2 = (tq_out2 - fq_out2_ck).detach().numpy()
maxdiff2 = np.max(np.abs(diff2))
coords2 = list(zip(*np.where(np.abs(diff2) == maxdiff2)))

tq_2_ops = get_operands_tq(coords2[0], tq_out1, tq_net[1])
fq_2_ops = get_operands_fq(coords2[0], fq_out1, fq_net[1], n_in=n_out1, m_in=m_out1, inq_layer=True)


# compare outputs of fake-quantized and outputs of true-quantized networks
import torch.nn as nn

net_cuda = net.to('cuda')

tq_full_a = nn.Sequential(*tq_net[:16]).to('cuda')
tq_full_b = nn.Sequential(*tq_net[16:]).to('cuda')  # I need to flatten 3D tensor to 1D vector

n_trials = 1000
images_idxs = torch.randperm(len(valid_set))[:n_trials]

net_correct_top1 = 0
net_correct_top5 = 0
tq_correct_top1 = 0
tq_correct_top5 = 0
match_top1 = 0
match_top5 = 0

for i, idx in enumerate(images_idxs):
    x, y = valid_set[idx][0].unsqueeze(0), valid_set[idx][1]

    fq_x = x.clone().to(torch.float32)

    net_out = net_cuda(fq_x.to('cuda'))
    net_preds = torch.topk(net_out, 5)[1]
    net_top1 = int(y == net_preds[..., 0])
    net_top5 = int(y in net_preds[:, ])

    tq_x = convert_input_image(fq_x.clone(), input_type)
    tq_x = tq_x.to(torch.float64)

    tq_out_a = tq_full_a(tq_x.to(torch.float64).to('cuda')).view(x.shape[0], 1, -1)
    tq_out = tq_full_b(tq_out_a)
    tq_preds = torch.topk(tq_out, 5)[1]
    tq_top1 = int(y == tq_preds[..., 0])
    tq_top5 = int(y in tq_preds[:, 0])

    net_correct_top1 += net_top1
    net_correct_top5 += net_top5
    tq_correct_top1 += tq_top1
    tq_correct_top5 += tq_top5
    match_top1 += int(net_top1 == tq_top1)
    match_top5 += int(net_top5 == tq_top5)

    if (i + 1) % 10 == 0:
        print("Iteration {}".format(i + 1))
        print("Fake-quantized: top1 {:7.3f}%, top5 {:7.3f}%".format(100 * net_correct_top1 / (i+1), 100 * net_correct_top5 / (i+1)))
        print("Net-quantized:  top1 {:7.3f}%, top5 {:7.3f}%".format(100 * tq_correct_top1 / (i+1), 100 * tq_correct_top5 / (i+1)))
        print("Agreement:      top1 {:7.3f}%, top5 {:7.3f}%".format(100 * match_top1 / (i+1), 100 * match_top5 / (i+1)))
