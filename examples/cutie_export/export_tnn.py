#
# export_tnn.py
#
# Author(s):
# Georg Rutishauser <georgr@iis.ee.ethz.ch>
# Joan Mihali
#
# Copyright (c) 2020-2021 ETH Zurich.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import argparse
import json
import importlib

import sys
from typing import Union
from pathlib import Path
from tqdm import tqdm

from copy import deepcopy

import torch
from torch import nn

#set the PYTHONPATH to include QuantLab's root directory
_QL_ROOTPATH = Path(__file__).absolute().parent.parent.parent

sys.path.append(str(_QL_ROOTPATH))


from systems.DVS128.dvs_cnn import DVSHybridNet, get_input_shape, load_data_set, DVSAugmentTransform


from systems.DVS128.utils.data.dvs128_dataset import DVS128DataSet
from quantlib.backends.cutie import convert_net, export_net

from test_gen.compute_DVSTNN import compute_tnn

_TOPOLOGY_PATH = _QL_ROOTPATH.joinpath('systems/DVS128/dvs_cnn')
_BATCH_SIZE_PER_GPU = 48

# the experiment config for the exp_id of the network specified by 'key'
def get_config(exp_id : int):
    config_filepath = _TOPOLOGY_PATH.joinpath(f'logs/exp{exp_id:04}/config.json')
    with open(config_filepath, 'r') as fp:
        config = json.load(fp)
    return config

def get_ckpt(exp_id : int, ckpt_id : Union[int, str], fold_id : int = 0):
    ckpt_str = f'epoch{ckpt_id:03}' if ckpt_id != -1 else 'best'
    ckpt_filepath = _TOPOLOGY_PATH.joinpath(f'logs/exp{exp_id:04}/fold{fold_id}/saves/{ckpt_str}.ckpt')
    load_opts = {}
    if not torch.cuda.is_available():
        load_opts['map_location'] = torch.device('cpu')
    return torch.load(ckpt_filepath, **load_opts)

def get_network(exp_id : int, ckpt_id : Union[int, str], fold_id : int = 0, quantized=False):
    cfg = get_config(exp_id)
    net_cfg = cfg['network']['kwargs']
    in_shape = get_input_shape(cfg)


    net = DVSHybridNet(**net_cfg)
    if not quantized:
        return net.eval()
    quant_fun_name = cfg['network']['quantize']['function']
    ctrl_fun_name = cfg['training']['quantize']['function']
    quant_cfg = cfg['network']['quantize']['kwargs']
    quant_module = importlib.import_module('.quantize', package='systems.DVS128.dvs_cnn')
    quant_func = getattr(quant_module, quant_fun_name)
    ctrl_func = getattr(quant_module, ctrl_fun_name)
    ctrl_cfg = cfg['training']['quantize']['kwargs']
    quant_net = quant_func(net, **quant_cfg)
    ckpt = get_ckpt(exp_id, ckpt_id, fold_id)
    state_dict = ckpt['net']
    # the checkpoint may be from a nn.DataParallel instance, so we need to
    # strip the 'module.' from all the keys
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.lstrip('module.'): v for k, v in state_dict.items()}
    quant_net.load_state_dict(state_dict)
    qctrls = ctrl_func(quant_net, **ctrl_cfg)
    for ctrl, sd in zip(qctrls, ckpt['qnt_ctrls']):
        ctrl.load_state_dict(sd)

    # we don't want to train this network anymore
    return quant_net.eval()


# get a validation dataset from the problem name.
def get_valid_dataset(cfg : dict, fold_id : int = 0):

    try:
        load_dataset_args = cfg['data']['valid']['dataset']['load_data_set']['kwargs']
    except KeyError:
        load_dataset_args = {}
    transform = DVSAugmentTransform
    try:
        transform_args = cfg['data']['valid']['dataset']['transform']['kwargs']
    except KeyError:
        transform_args = {}

    n_folds = cfg['data']['cv']['n_folds']
    cnn_win = load_dataset_args['cnn_win']
    transform_args['cnn_window'] = cnn_win
    transform_inst = transform(**transform_args)
    path_data = _QL_ROOTPATH.joinpath('systems').joinpath('DVS128/data')
    return load_data_set(partition='valid', path_data=str(path_data), n_folds=n_folds, current_fold_id=fold_id, cv_seed=0, transform=transform_inst, **load_dataset_args)


def get_dataloader(cfg : dict):

    if torch.cuda.is_available():
        bs = torch.cuda.device_count() * _BATCH_SIZE_PER_GPU
    else:
        # network will be executed on CPU (not recommended!!)
        bs = 16
    ds = get_valid_dataset(cfg)
    return torch.utils.data.DataLoader(ds, bs)

def validate(net : nn.Module, dl : torch.utils.data.DataLoader, print_interval : int = 10, argmax_offsets : torch.Tensor = None, eps_w : torch.Tensor = None):
    net = net.eval()
    # we assume that the net is on CPU as this is required for some
    # integerization passes
    if torch.cuda.is_available():
        net = net.to('cuda')
        device = 'cuda'
        if torch.cuda.device_count() != 1:
            net = nn.DataParallel(net)
    else:
        device = 'cpu'

    n_tot = 0
    n_correct = 0

    for i, (xb, yb) in enumerate(tqdm(dl)):
        yn = net(xb.to(device))
        yn = yn.to('cpu').detach().squeeze()
        if argmax_offsets is not None:
            yn += argmax_offsets[None, :]
        if eps_w is not None:
            yn *= eps_w.reshape((1,-1))
        n_tot += xb.shape[0]
        n_correct += (yn.argmax(dim=1) == yb).sum()
        if ((i+1)%print_interval == 0):
            print(f'Accuracy after {i+1} batches: {n_correct/n_tot}')

    print(f'Final accuracy: {n_correct/n_tot}')
    net.to('cpu')


def compare_nets(fqnet, intnet, indata, offsets=None):

    if len(indata.shape) != 4:
        indata = indata.unsqueeze(0)
    in_windows = torch.split(indata, 4, dim=1)
    cnn_outs_fq = []
    cnn_outs_tq = []
    for i,win in enumerate(in_windows):
        ad_fq = fqnet.cnn.adapter
        ad_int = intnet.cnn.adapter
        ad_out_fq = ad_fq(win)
        ad_out_int = ad_int(win)
        ad_out_int_fq = (ad_out_int + 1) * ad_fq[-1].get_eps()
        if not torch.all(ad_out_int_fq == ad_out_fq):
            print(f"failure in adapter, iteration {i}")
        f0_fq = fqnet.cnn.features[:4]
        f0_int = intnet.cnn.features[:4]

        f0_out_fq = f0_fq(ad_out_fq)
        f0_out_int = f0_int(ad_out_int)
        f0_out_intfq = (f0_out_int + 1) * f0_fq[-1].get_eps()
        if not torch.all(f0_out_intfq == f0_out_fq):
            print(f"failure in f0, iteration {i}")

        f_int = intnet.cnn.features
        f_fq = fqnet.cnn.features
        f_out_int = f_int(ad_out_int)
        f_out_fq = f_fq(ad_out_fq)
        f_out_intfq = (f_out_int+1)*f_fq[-1].get_eps()
        if not torch.all(f_out_intfq == f_out_fq):
            print(f"failure in features, iteration {i}")
        cnn_outs_fq.append(f_out_fq.flatten(start_dim=1))
        cnn_outs_tq.append(f_out_int.flatten(start_dim=1))


    tcn_in_fq = torch.stack(cnn_outs_fq, dim=2)
    tcn_in_int = torch.stack(cnn_outs_tq, dim=2)
    tcn_f_fq = fqnet.tcn.features
    tcn_f_int = intnet.tcn.features
    tcn_fout_fq = tcn_f_fq(tcn_in_fq)
    tcn_fout_int = tcn_f_int(tcn_in_int)
    tcn_fout_intfq = (tcn_fout_int+1)*tcn_f_fq[-1].get_eps()
    tcn_out_fq = fqnet.tcn.classifier(tcn_fout_fq)
    tcn_out_int = intnet.tcn.classifier(tcn_fout_int)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TNN Exporter for CUTIE')
    parser.add_argument('--exp_id', required=True,
                        help='Experiment to integerize and export. The specified experiment must be a for a PACT/TQT-quantized network! Can be "best" or the index of the checkpoint')
    parser.add_argument('--validate_fq', action='store_true',
                        help='Whether to validate the fake-quantized network on the appropriate dataset')
    parser.add_argument('--validate_tq', action='store_true',
                        help='Whether to validate the integerized network on the appropriate dataset')
    parser.add_argument('--export_dir', type=str, default=None,
                        help='Export the integerized network to the specified directory.')

    parser.add_argument('--c_out_dir', '-c', type=str, default=None, help='Where to export the Kraken C application. If not provided, it will go to a subdirectory "application" in "export_dir"')
    parser.add_argument('--fold_id', type=int, default=0)
    parser.add_argument('--ckpt_id', required=True, type=int,
                        help='Checkpoint to integerize and export. The specified checkpoint must be fully quantized with PACT/TQT!')
    parser.add_argument('--n_windows', '-n', type=int, default=None)

    args = parser.parse_args()

    name = f'DVSTNN_exp{args.exp_id}_ckpt{args.ckpt_id}'

    exp_id = int(args.exp_id) if args.exp_id.isnumeric() else args.exp_id
    print(f'Loading DVSHybridNet, experiment {exp_id}, checkpoint {args.ckpt_id}')

    qnet = get_network(exp_id, args.ckpt_id, fold_id=args.fold_id, quantized=True)
    qnet_copy = get_network(exp_id, args.ckpt_id, fold_id=args.fold_id, quantized=True)
    exp_cfg = get_config(exp_id)
    dl = get_dataloader(exp_cfg)
    if args.validate_fq:
        print(f'Validating fake-quantized TNN on DVS128 Dataset')
        validate(qnet, dl, 1)
    print('Integerizing TNN')
    int_net, argmax_offsets, eps_w = convert_net(qnet, get_input_shape(exp_cfg), False, True)
    in_data = dl.dataset[42][0]
    compare_nets(qnet_copy.cpu(), int_net.cpu(), in_data.cpu(), argmax_offsets)
    if args.validate_tq:
        print(f'Validating true-quantized TNN on DVS128 Dataset')
        validate(int_net, dl, 1, argmax_offsets, eps_w)

    print(in_data.requires_grad)


    export_net(int_net.cpu(), args.export_dir, in_data, name, split_input=exp_cfg['network']['kwargs']['cnn_window'])

    c_out_path = Path(args.c_out_dir).resolve() if args.c_out_dir is not None else Path(args.export_dir).joinpath('application').resolve()
    c_out_path.mkdir(parents=True, exist_ok=True)

    compute_tnn(n_layers=0, in_directory=args.export_dir, in_prefix=name, out_directory=c_out_path, n_windows=args.n_windows)

