# 
# integerize_pactnets.py
# 
# Author(s):
# Georg Rutishauser <georgr@iis.ee.ethz.ch>
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

from collections import namedtuple
from functools import partial
from typing import Union, Optional
from tqdm import tqdm
import json
import torch
from torch import nn, fx


# set the PYTHONPATH to include QuantLab's root directory
import sys
from pathlib import Path
_QL_ROOTPATH = Path(__file__).absolute().parent.parent.parent

sys.path.append(str(_QL_ROOTPATH))

# import the get_dataset functions for CIFAR10 and ImageNet
from systems.CIFAR10.utils.data import load_data_set as load_cifar10
from systems.CIFAR10.utils.transforms import CIFAR10PACTQuantTransform
from systems.CIFAR10.utils.transforms.transforms import CIFAR10STATS
from systems.ILSVRC12.utils.data import load_ilsvrc12
from systems.ILSVRC12.utils.transforms import ILSVRC12PACTQuantTransform
from systems.ILSVRC12.utils.transforms.transforms import ILSVRC12STATS

_CIFAR10_EPS = CIFAR10STATS['quantize']['eps']
_ILSVRC12_EPS = ILSVRC12STATS['quantize']['eps']

# import the networks
from systems.CIFAR10.VGG import VGG
from systems.ILSVRC12.MobileNetV1 import MobileNetV1
from systems.ILSVRC12.MobileNetV2 import MobileNetV2
from systems.ILSVRC12.MobileNetV3 import MobileNetV3

# import the quantization functions for the networks
from systems.CIFAR10.VGG.quantize import pact_recipe as quantize_vgg, get_pact_controllers as controllers_vgg
from systems.ILSVRC12.MobileNetV1.quantize import pact_recipe as quantize_mnv1, get_pact_controllers as controllers_mnv1
from systems.ILSVRC12.MobileNetV2.quantize import pact_recipe as quantize_mnv2, get_pact_controllers as controllers_mnv2
from systems.ILSVRC12.MobileNetV3.quantize import pact_recipe as quantize_mnv3, get_pact_controllers as controllers_mnv3

# import the DORY backend
from quantlib.backends.dory import export_net, DORYHarmonizePass
# import the PACT/TQT integerization pass
from quantlib.editing.fx.passes.pact import IntegerizePACTNetPass
from quantlib.editing.fx.util import module_of_node
from quantlib.algorithms.pact.pact_ops import *
# organize quantization functions, datasets and transforms by network
QuantUtil = namedtuple('QuantUtil', 'problem quantize get_controllers network in_shape eps_in D bs')

# get a validation dataset from the problem name.
def get_valid_dataset(problem : str, quantize : str, pad_img : Optional[int] = None, clip : bool = False):
    load_dataset_fn, transform = (load_cifar10, CIFAR10PACTQuantTransform) if problem == 'CIFAR10' else (load_ilsvrc12, ILSVRC12PACTQuantTransform)
    transform_inst = transform(augment=False, quantize=quantize, n_q=256, pad_channels=pad_img, clip=clip)
    path_data = _QL_ROOTPATH.joinpath('systems').joinpath(problem).joinpath('data')
    return load_dataset_fn(partition='valid', path_data=str(path_data), n_folds=1, current_fold_id=0, cv_seed=0, transform=transform_inst)

# the QuantLab problem being solved by the specified network.
def get_system(key : str):
    return 'CIFAR10' if key == 'VGG' else 'ILSVRC12'

# the topology directory where the specified network is defined
def get_topology_dir(key : str):
    return _QL_ROOTPATH.joinpath('systems').joinpath(get_system(key)).joinpath(key)

# batch size is per device, determined on Nvidia RTX2080. You may have to change
# this if you have different GPUs
_QUANT_UTILS = {
    'VGG': QuantUtil(problem='CIFAR10', quantize=quantize_vgg, get_controllers=controllers_vgg, network=VGG, in_shape=(1,3,32,32), eps_in=_CIFAR10_EPS, D=2**19, bs=256),
    'MobileNetV1': QuantUtil(problem='ILSVRC12', quantize=quantize_mnv1, get_controllers=controllers_mnv1, network=MobileNetV1, in_shape=(1,3,224,224), eps_in=_ILSVRC12_EPS, D=2**19, bs=96),
    'MobileNetV2': QuantUtil(problem='ILSVRC12', quantize=quantize_mnv2, get_controllers=controllers_mnv2, network=MobileNetV2, in_shape=(1,3,224,224), eps_in=_ILSVRC12_EPS, D=2**19, bs=53),
    'MobileNetV3': QuantUtil(problem='ILSVRC12', quantize=quantize_mnv3, get_controllers=controllers_mnv3, network=MobileNetV3, in_shape=(1,3,224,224), eps_in=_ILSVRC12_EPS, D=2**19, bs=53),
}

# the experiment config for the exp_id of the network specified by 'key'
def get_config(key : str, exp_id : int):
    config_filepath = get_topology_dir(key).joinpath(f'logs/exp{exp_id:04}/config.json')
    with open(config_filepath, 'r') as fp:
        config = json.load(fp)
    return config

def get_ckpt(key : str, exp_id : int, ckpt_id : Union[int, str]):
    ckpt_str = f'epoch{ckpt_id:03}' if ckpt_id != -1 else 'best'
    ckpt_filepath = get_topology_dir(key).joinpath(f'logs/exp{exp_id:04}/fold0/saves/{ckpt_str}.ckpt')
    return torch.load(ckpt_filepath)

def get_network(key : str, exp_id : int, ckpt_id : Union[int, str], quantized=False):
    cfg = get_config(key, exp_id)
    qu = _QUANT_UTILS[key]
    quant_cfg = cfg['network']['quantize']['kwargs']
    ctrl_cfg = cfg['training']['quantize']['kwargs']
    net_cfg = cfg['network']['kwargs']
    net = qu.network(**net_cfg)
    if not quantized:
        return net.eval()
    quant_net = qu.quantize(net, **quant_cfg)
    ckpt = get_ckpt(key, exp_id, ckpt_id)
    state_dict = ckpt['net']
    # the checkpoint may be from a nn.DataParallel instance, so we need to
    # strip the 'module.' from all the keys
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.lstrip('module.'): v for k, v in state_dict.items()}
    quant_net.load_state_dict(state_dict)
    qctrls = qu.get_controllers(quant_net, **ctrl_cfg)
    for ctrl, sd in zip(qctrls, ckpt['qnt_ctrls']):
        ctrl.load_state_dict(sd)


    # we don't want to train this network anymore
    return quant_net.eval()

def get_dataloader(key : str, quantize : str, pad_img : Optional[int] = None, clip : bool = False):
    qu = _QUANT_UTILS[key]
    if torch.cuda.is_available():
        bs = torch.cuda.device_count() * qu.bs
    else:
        # network will be executed on CPU (not recommended!!)
        bs = 16
    ds = get_valid_dataset(qu.problem, quantize, pad_img=pad_img, clip=clip)
    return torch.utils.data.DataLoader(ds, bs)


def validate(net : nn.Module, dl : torch.utils.data.DataLoader, print_interval : int = 10):
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
        n_tot += xb.shape[0]
        n_correct += (yn.to('cpu').argmax(dim=1) == yb).sum()
        if ((i+1)%10 == 0):
            print(f'Accuracy after {i+1} batches: {n_correct/n_tot}')

    print(f'Final accuracy: {n_correct/n_tot}')
    net.to('cpu')


def get_input_channels(net : fx.GraphModule):
    for node in net.graph.nodes:
        if node.op == 'call_module' and isinstance(module_of_node(net, node), (nn.Conv1d, nn.Conv2d)):
            conv = module_of_node(net, node)
            return conv.in_channels

# THIS IS WHERE THE BUSINESS HAPPENS!
def integerize_network(net : nn.Module, key : str, fix_channels : bool, dory_harmonize : bool):
    qu = _QUANT_UTILS[key]
    # All we need to do to integerize a fake-quantized network is to run the
    # IntegerizePACTNetPass on it! Afterwards, the ONNX graph it produces will
    # contain only integer operations. Any divisions in the integerized graph
    # will be by powers of 2 and can be implemented as bit shifts.
    in_shp = qu.in_shape
    int_pass = IntegerizePACTNetPass(shape_in=in_shp, eps_in=qu.eps_in, D=qu.D, fix_channel_numbers=fix_channels)
    int_net = int_pass(net)
    if fix_channels:
        # we may have modified the # of input channels so we need to adjust the
        # input shape
        in_shp_l = list(in_shp)
        in_shp_l[1] = get_input_channels(int_net)
        in_shp = tuple(in_shp_l)
    if dory_harmonize:
        # the DORY harmonization pass:
        # - wraps and aligns averagePool nodes so
        #   they behave as they do in the PULP-NN kernel
        # - replaces quantized adders with DORYAdder modules which are exported
        #   as custom "QuantAdd" ONNX nodes
        dory_harmonize_pass = DORYHarmonizePass(in_shape=in_shp)
        int_net = dory_harmonize_pass(int_net)

    return int_net

def export_integerized_network(net : nn.Module, key : str, export_dir : str, name : str, in_idx : int = 42, pad_img : Optional[int] = None, clip : bool = False):
    qu = _QUANT_UTILS[key]
    # use a real image from the validation set
    ds = get_valid_dataset(get_system(key), quantize='int', pad_img=pad_img, clip=clip)
    test_input = ds[in_idx][0].unsqueeze(0)
    export_net(net, name=name, out_dir=export_dir, eps_in=qu.eps_in, integerize=False, D=qu.D, in_data=test_input)

def export_unquant_net(net : nn.Module, key : str, export_dir : str, name : str):
    out_path = Path(export_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    onnx_file = f"{name}.onnx"
    onnx_path = out_path.joinpath(onnx_file)
    ds = get_valid_dataset(get_system(key), quantize='none')
    test_input = ds[42][0].unsqueeze(0)
    torch.onnx.export(net.to('cpu'),
                      test_input,
                      str(onnx_path),
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True)



# if this script is executed directly, expose all the above functions as
# command line arguments.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QuantLab Example Flows')
    parser.add_argument('--net', required=True, type=str,
                        help='Network to treat - can be "MobileNetV1", "MobileNetV2", MobileNetV3 or "VGG"')
    parser.add_argument('--exp_id', required=True,
                        help='Experiment to integerize and export. The specified experiment must be a for a PACT/TQT-quantized network! Can be "best" or the index of the checkpoint')
    parser.add_argument('--ckpt_id', required=True, type=int,
                        help='Checkpoint to integerize and export. The specified checkpoint must be fully quantized with PACT/TQT!')
    parser.add_argument('--fix_channels', action='store_true',
                        help='Fix channels of conv layers for compatibility with DORY')
    parser.add_argument('--clip_inputs', action='store_true',
                        help='ghettofix to clip inputs to be unsigned')
    parser.add_argument('--validate_fq', action='store_true',
                        help='Whether to validate the fake-quantized network on the appropriate dataset')
    parser.add_argument('--validate_tq', action='store_true',
                        help='Whether to validate the integerized network on the appropriate dataset')
    parser.add_argument('--export_unquant', action='store_true',
                        help='Also export the unquantized network')
    parser.add_argument('--export_dir', type=str, default=None,
                        help='Export the integerized network to the specified directory.')
    parser.add_argument('--export_name', type=str, default=None,
                        help='Name of the exported ONNX graph. By default, this is identical to the value of the "--net" flag')
    parser.add_argument('--accuracy_print_interval', type=int, default=10,
                        help='While evaluating networks on the validation set, print the intermediate accuracy every N batches')
    parser.add_argument('--no_dory_harmonize', action='store_true',
                        help='If supplied, don\'t align averagePool nodes\' associated requantization nodes and replace adders with DORYAdders')



    args = parser.parse_args()

    if args.export_dir is not None:
        export_name = args.net if args.export_name is None else args.export_name

    exp_id = int(args.exp_id) if args.exp_id.isnumeric() else args.exp_id

    print(f'Loading network {args.net}, experiment {exp_id}, checkpoint {args.ckpt_id}')
    qnet = get_network(args.net, exp_id, args.ckpt_id, quantized=True)

    if args.validate_fq:
        print(f'Validating fake-quantized network {args.net} on dataset {get_system(args.net)}')
        dl = get_dataloader(args.net, quantize='fake')
        validate(qnet, dl, args.accuracy_print_interval)

    print(f'Integerizing network {args.net}')

    int_net = integerize_network(qnet, args.net, args.fix_channels, not args.no_dory_harmonize)

    if args.fix_channels:
        pad_img = get_input_channels(int_net)
    else:
        pad_img = None

    if args.validate_tq:
        dl = get_dataloader(args.net, quantize='int', pad_img=pad_img)
        validate(int_net, dl, args.accuracy_print_interval)

    if args.export_dir is not None:
        print(f'Exporting integerized network {args.net} to directory {args.export_dir} under name {export_name}')
        export_integerized_network(int_net, args.net, args.export_dir, export_name, pad_img=pad_img, clip=args.clip_inputs)
        if args.export_unquant:
            net_unq = get_network(args.net, exp_id, args.ckpt_id, quantized=False)
            export_unquant_net(net_unq, args.net, args.export_dir, export_name)

