import os
import sys

import math
import json
import importlib
import torch
from torch import nn
import numpy as np

import quantlib.graphs as qg
import quantlib.graphs.analyse as qa
import quantlib.graphs.edit as qe
from quantlib.graphs.analyse import Node

from backends.twn_accelerator.compiler_vgg import compile_vgg

from quantlib.graphs import Morpher, ScopeRule
from quantlib.graphs.graphs import add_ste_tunnels, add_linear_tunnels, add_output_tunnel, remove_tunnels

sys.path.insert(0, os.pardir)  # make QuantLab packages accessible
sys.path.append('./pydevd-pycharm.egg')

#import pydevd_pycharm
#pydevd_pycharm.settrace('vilan2', port=9002, stdoutToServer=True, stderrToServer=True)



__MASTER_PROC_RANK__ = 0
__MAX_EXPERIMENTS__  = 1000
__ALIGN_EXP__        = math.ceil(math.log10(__MAX_EXPERIMENTS__))  # experiment ID string length (decimal literal)
__MAX_CV_FOLDS__     = 10
__ALIGN_CV_FOLDS__   = math.ceil(math.log10(__MAX_CV_FOLDS__))  # cross-validation fold ID string length (decimal literal)
__EVAL__             = True # whether to check the accuracy of the exported/quantized net


with open('backends/twn_accelerator/source.json') as fp:
    specs = json.load(fp)


class MiniLogbook(object):
    def __init__(self, specs):
        self.lib = importlib.import_module('.'.join(['systems', specs['problem'], specs['topology']]))
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

# in version 1.5.0 (maybe others too), the torch jit tracer has buggy behavior when encountering Dropout nodes,
# so we remove them here
net_nodes = qa.list_nodes(net)
dropout_nodes = qa.find_nodes(net_nodes, [qa.rule_dropout_nodes], 'and')
for n in dropout_nodes:
    qe._replace_node(net, n.name, nn.Identity())

morpher = Morpher(net, (torch.ones(1, 3, 224, 224),))
P = morpher.get_opgraph(morpher.get_pytorch_graph())
Q = add_output_tunnel(add_linear_tunnels(add_ste_tunnels(P)), 'classifier.6')

H2Dtemplate = morpher.get_template(Q, 'adapter.0', 'features.0.ste.tunnel.0', include_interface=True)
rho_H2D = ScopeRule(H2Dtemplate, set(['features.0.ste.tunnel.0']))

#with MaxPool
D2Dtemplate_1 = morpher.get_template(Q, 'features.0.ste.tunnel.0', 'features.4.ste.tunnel.0', include_interface=True)
rho_D2D1 = ScopeRule(D2Dtemplate_1, set(['features.0.ste.tunnel.0', 'features.4.ste.tunnel.0']))

#without MaxPool
D2Dtemplate_2 = morpher.get_template(Q, 'features.4.ste.tunnel.0', 'features.7.ste.tunnel.0', include_interface=True)
rho_D2D2 = ScopeRule(D2Dtemplate_2, set(['features.4.ste.tunnel.0', 'features.7.ste.tunnel.0']))

D2Htemplate = morpher.get_template(Q, 'features.46.ste.tunnel.0', 'avgpool.tunnel.0', include_interface=True)
rho_D2H = ScopeRule(D2Htemplate, set(['features.46.ste.tunnel.0', 'avgpool.tunnel.0']))

# with a ReLU
H2Htemplate_1 = morpher.get_template(Q, 'avgpool.tunnel.0', 'classifier.1.tunnel.0', include_interface=True)
rho_H2H1 = ScopeRule(H2Htemplate_1, set(['avgpool.tunnel.0', 'classifier.1.tunnel.0']))
# without a ReLU (i.e. only the last layer)
H2Htemplate_2 = morpher.get_template(Q, 'classifier.4.tunnel.0', '__output_tunnel', include_interface=True)
rho_H2H2 = ScopeRule(H2Htemplate_2, set(['classifier.4.tunnel.0', '__output_tunnel']))


for i, g in enumerate(rho_H2D.discover(Q)):
    Q = rho_H2D.apply(Q, g, '_'.join(['layer_h2d', str(i)]))

for i, g in enumerate(rho_D2D1.discover(Q)):
    Q = rho_D2D1.apply(Q, g, '_'.join(['layer_d2d', str(i)]))

for j, g in enumerate(rho_D2D2.discover(Q)):
    Q = rho_D2D2.apply(Q, g, '_'.join(['layer_d2d', str(i + 1 + j)]))

for i, g in enumerate(rho_D2H.discover(Q)):
    Q = rho_D2H.apply(Q, g, '_'.join(['layer_d2h', str(i)]))

for i, g in enumerate(rho_H2H1.discover(Q)):
    Q = rho_H2H1.apply(Q, g, '_'.join(['layer_h2h', str(i)]))

for j, g in enumerate(rho_H2H2.discover(Q)):
    Q = rho_H2H2.apply(Q, g, '_'.join(['layer_h2h', str(i + 1 + j)]))

R = remove_tunnels(Q)

import networkx as nx

M = morpher.get_pytorch_graph()
kpart = nx.bipartite.sets(M)[0]
mpart = nx.bipartite.sets(M)[1]
mapping = {n.split('/', 1)[-1]: n for n in R.nodes if n.split('/', 1)[-1] in M.nodes}
N = nx.relabel_nodes(M, mapping, copy=True)

op_2_scope = {v: v.split('/', 1)[0] for v in mapping.values()}
data_2_scope = {n: n for n in mpart}
O = morpher.get_view(N, op_2_scope, data_2_scope)
W = morpher.get_opgraph(O)
old_2_new = dict()
for j, n in enumerate(nx.algorithms.dag.topological_sort(W)):
    old_name = n
    new_name = '_'.join([*n.split('_')[:-1], str(j)])
    old_2_new[old_name] = new_name

layer_name_2_pytorch_modules = dict()
for k, v in old_2_new.items():
    T = R.subgraph([n for n in R.nodes if n.split('/', 1)[0] == k])
    layer_name_2_pytorch_modules[v] = [R.nodes[n]['pytorch'] for n in nx.algorithms.dag.topological_sort(T)]




# # compile VGG
def apply_ste_postproc(x, ste_n, ste_m):
    ex = (2 * ste_m) / (ste_n - 1)
    return (x * ex).to(torch.float32)


def revert_ste_postproc(x, ste_n, ste_m):
    ex = (2 * ste_m) / (ste_n - 1)
    return (x / ex).to(torch.float64).round()


def convert_input_image(img, input_type):
    """Rescale a normalised ILSVRC12 data point to UINT8 or INT8 range."""
    if not ('int' in input_type):
        return img
    else:
        from systems.ILSVRC12.VGG.preprocess import _ImageNet
        mean = torch.tensor(_ImageNet['Normalize']['mean']) * 255.
        std = torch.tensor(_ImageNet['Normalize']['std']) * 255.

        new_img = img.squeeze(0)
        new_img = (new_img.permute(1, 2, 0) * std + mean).permute(2, 0, 1).clamp(min=0., max=255.).round()
        new_img = new_img.unsqueeze(0)

        if input_type == 'int8':  # signed integer
            new_img -= 128.

    return new_img

#valid values:
# 'int8', 'uint8' or 'float'
input_type = 'float'
output_dir = 'trialVGG'
#tq_net, fq_net, export_net, data_out_dir = compile_vgg(layer_name_2_pytorch_modules, output_dir=output_dir, input_type=input_type)
tq_net, fq_net, data_out_dir = compile_vgg(layer_name_2_pytorch_modules, output_dir=output_dir, input_type=input_type)

train_set, valid_set = logbook.lib.load_data_sets(logbook)
# dump input and expected output of true-quantized net
tq_x_in_fp32 = valid_set[0][0].unsqueeze(0).to(torch.float32)
tq_x_in_fp32_conv = convert_input_image(tq_x_in_fp32, input_type)
# convert to nhwc
tq_x_in_fp32_conv_nhwc = tq_x_in_fp32.permute(0, 2, 3, 1)



with open(os.path.join(data_out_dir, 'test_input'), 'wb') as fh:
    for el in tq_x_in_fp32_conv_nhwc.numpy().flatten():
        fh.write(el)


# convert the lists of layers to sequential nets

export_net_fp32 = nn.Sequential(*export_net).to(torch.float32)
export_net_fp32.eval()

with torch.no_grad():
    exp_out = export_net_fp32(tq_x_in_fp32_conv)


with open(os.path.join(data_out_dir, 'exp_output'), 'wb') as fh:
    for el in exp_out.detach().numpy().flatten():
        fh.write(el)



# get all the layers out of the original net
orig_layers_conv = [n.module for n in qa.list_nodes(net.adapter)]
orig_layers_conv += [n.module for n in qa.list_nodes(net.features)]
orig_layers_conv.append(net.avgpool)
# need to add the flatten layer because it is added in compiler_vgg
orig_layers_conv.append(nn.Flatten())
orig_layers = orig_layers_conv + [n.module for n in qa.list_nodes(net.classifier)]

# group the original network into the same blocks as fq_net
fq_lens = [len(mod) for mod in fq_net]
def get_sublists(l : list, lens : list):
    assert len(l) == sum(lens), "sum of lengths must be equal to length of list"
    out = []
    idx = 0
    for ll in lens:
        out.append(l[idx:idx+ll])
        idx += ll
    return out

orig_layers_seq = [nn.Sequential(*m) for m in get_sublists(orig_layers, fq_lens)]


if __EVAL__:
    def compare_subtensor(t1 : torch.Tensor, t2 : torch.Tensor):
        # compare two tensors on their overlaps. returns sum of absolute differences.
        assert len(t1.shape) == len(t2.shape), "tensors to compare must have the same number of dimensions"
        overlap_shape = tuple(min(s1, s2) for s1, s2 in zip(t1.shape, t2.shape))
        return torch.sum(torch.abs(t1[[slice(0, s) for s in overlap_shape]] - t2[[slice(0, s) for s in overlap_shape]]))
    n_trials = 1
    match = 0
    for i in range(n_trials):
        # i = torch.randint(low=0, high=len(valid_set), size=(1,))
        img = valid_set[i][0].unsqueeze(0)

        fq_x_in = img.clone().to(torch.float32)

        fq_x = fq_x_in.clone()

        tq_x_in_fp32 = convert_input_image(fq_x_in.clone(), input_type)
        tq_x_in = tq_x_in_fp32.to(torch.float64)
        tq_x = tq_x_in.clone()
        orig_x = fq_x_in.clone()

        errors = []
        for l, (tql, fql, orig_l) in enumerate(zip(tq_net, fq_net, orig_layers_seq)):

            tql = tql.to(torch.float64)
            fql = fql.to(torch.float32)
            tql.eval()
            orig_l = orig_l.to(torch.float32)

            tq_x = tql(tq_x)

            fq_x = fql(fq_x)

            orig_x = orig_l(orig_x)


            if l < 15:
                ste = fql[-1]
                fq_x_ck = revert_ste_postproc(fq_x, ste.num_levels, ste.abs_max_value)
                orig_x_ck = revert_ste_postproc(orig_x, ste.num_levels, ste.abs_max_value)
                diff = (tq_x - fq_x_ck).detach().numpy()
                diff_orig = compare_subtensor(orig_x_ck, fq_x_ck)
                print("difference between original net and fq net returned by compile_vgg in layer {}: {}".format(l, diff_orig))
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
