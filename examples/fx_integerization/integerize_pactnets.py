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

from dataclasses import dataclass, field
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
from systems.MNIST.utils.data import load_data_set as load_mnist
from systems.MNIST.utils.transforms import MNISTPACTQuantTransform
from systems.MNIST.utils.transforms.transforms import MNISTSTATS
from systems.ILSVRC12.utils.data import load_ilsvrc12
from systems.ILSVRC12.utils.transforms import ILSVRC12PACTQuantTransform
from systems.ILSVRC12.utils.transforms.transforms import ILSVRC12STATS
from systems.DVS128.dvs_cnn.preprocess import load_data_set as load_dvs128, DVSAugmentTransform


_CIFAR10_EPS = CIFAR10STATS['quantize']['eps']
_ILSVRC12_EPS = ILSVRC12STATS['quantize']['eps']
_MNIST_EPS = MNISTSTATS['quantize']['eps']

# import the networks
from systems.CIFAR10.VGG import VGG
from systems.CIFAR10.ResNet import ResNet as ResNetCIFAR
from systems.ILSVRC12.MobileNetV1 import MobileNetV1
from systems.ILSVRC12.MobileNetV2 import MobileNetV2
from systems.ILSVRC12.MobileNetV3 import MobileNetV3
from systems.ILSVRC12.ResNet import ResNet
from systems.DVS128.dvs_cnn import DVSHybridNet, get_input_shape as get_in_shape_dvsnet

# import the quantization functions for the networks
from systems.CIFAR10.VGG.quantize import pact_recipe as quantize_vgg, get_pact_controllers as controllers_vgg
from systems.CIFAR10.ResNet.quantize import pact_recipe as quantize_resnet_cifar, get_pact_controllers as controllers_resnet_cifar
from systems.ILSVRC12.MobileNetV1.quantize import pact_recipe as quantize_mnv1, get_pact_controllers as controllers_mnv1
from systems.ILSVRC12.MobileNetV2.quantize import pact_recipe as quantize_mnv2, get_pact_controllers as controllers_mnv2
from systems.ILSVRC12.MobileNetV3.quantize import pact_recipe as quantize_mnv3, get_pact_controllers as controllers_mnv3
from systems.ILSVRC12.ResNet.quantize import pact_recipe as quantize_resnet, get_pact_controllers as controllers_resnet
from systems.DVS128.dvs_cnn.quantize import pact_recipe as quantize_dvsnet, get_pact_controllers as controllers_dvsnet
from systems.MNIST.simpleCNN import simpleCNN
from systems.MNIST.simpleCNN.quantize import pact_recipe as quantize_simpleCNN, get_pact_controllers as controllers_simpleCNN

# import the DORY backend
from quantlib.backends.dory import export_net, export_dvsnet, DORYHarmonizePass
# import the PACT/TQT integerization pass
from quantlib.editing.fx.passes.pact import IntegerizePACTNetPass
from quantlib.editing.fx.util import module_of_node
from quantlib.algorithms.pact.pact_ops import *
# organize quantization functions, datasets and transforms by network
@dataclass
class QuantUtil:
    problem : str
    topo : str
    quantize : callable
    get_controllers : callable
    network : type
    in_shape : tuple
    eps_in : float
    D : int
    bs : int
    get_in_shape : callable
    load_dataset_fn : callable
    transform : type
    n_levels_in : int
    export_fn : callable
    code_size : int
    network_args : dict = field(default_factory=dict)
    quant_transform_args : dict = field(default_factory=dict)
#QuantUtil = namedtuple('QuantUtil', 'problem quantize get_controllers network in_shape eps_in D bs get_in_shape load_dataset_fn transform quant_transform_args n_levels_in export_fn')

# get a validation dataset from the problem name.
def get_valid_dataset(key : str, cfg : dict, quantize : str, pad_img : Optional[int] = None, clip : bool = False):
    qu = _QUANT_UTILS[key]
    load_dataset_fn = qu.load_dataset_fn
    try:
        load_dataset_args = cfg['data']['valid']['dataset']['load_data_set']['kwargs']
    except KeyError:
        load_dataset_args = {}
    transform = qu.transform
    try:
        transform_args = cfg['data']['valid']['dataset']['transform']['kwargs']
    except KeyError:
        transform_args = {}
    transform_args.update(qu.quant_transform_args)
    if key == 'dvs_cnn':
        cnn_win = load_dataset_args['cnn_win']
        transform_args['cnn_window'] = cnn_win
    transform_inst = transform(quantize=quantize, pad_channels=pad_img, clip=clip, **transform_args)
    path_data = _QL_ROOTPATH.joinpath('systems').joinpath(qu.problem).joinpath('data')
    return load_dataset_fn(partition='valid', path_data=str(path_data), n_folds=1, current_fold_id=0, cv_seed=0, transform=transform_inst, **load_dataset_args)



# batch size is per device, determined on Nvidia RTX2080. You may have to change
# this if you have different GPUs
_QUANT_UTILS = {
    'VGG': QuantUtil(problem='CIFAR10', topo='VGG', quantize=quantize_vgg, get_controllers=controllers_vgg, network=VGG, in_shape=(1,3,32,32), eps_in=_CIFAR10_EPS, D=2**19, bs=256, get_in_shape=None, load_dataset_fn=load_cifar10, transform=CIFAR10PACTQuantTransform, quant_transform_args={'n_q':256}, n_levels_in=256, export_fn=export_net, code_size=150000),
    'MobileNetV1': QuantUtil(problem='ILSVRC12', topo='MobileNetV1', quantize=quantize_mnv1, get_controllers=controllers_mnv1, network=MobileNetV1, in_shape=(1,3,224,224), eps_in=_ILSVRC12_EPS, D=2**19, bs=96, get_in_shape=None, load_dataset_fn=load_ilsvrc12, transform=ILSVRC12PACTQuantTransform, quant_transform_args={'n_q':256}, n_levels_in=256, export_fn=export_net, code_size=135000),
    'MobileNetV2': QuantUtil(problem='ILSVRC12', topo='MobileNetV2', quantize=quantize_mnv2, get_controllers=controllers_mnv2, network=MobileNetV2, in_shape=(1,3,224,224), eps_in=_ILSVRC12_EPS, D=2**19, bs=53, get_in_shape=None, load_dataset_fn=load_ilsvrc12, transform=ILSVRC12PACTQuantTransform, quant_transform_args={'n_q':256}, n_levels_in=256, export_fn=export_net, code_size=150000),
    'MobileNetV3': QuantUtil(problem='ILSVRC12', topo='MobileNetV3', quantize=quantize_mnv3, get_controllers=controllers_mnv3, network=MobileNetV3, in_shape=(1,3,224,224), eps_in=_ILSVRC12_EPS, D=2**19, bs=53, get_in_shape=None, load_dataset_fn=load_ilsvrc12, transform=ILSVRC12PACTQuantTransform, quant_transform_args={'n_q':256}, n_levels_in=256, export_fn=export_net, code_size=150000),
    'ResNet': QuantUtil(problem='ILSVRC12', topo='ResNet', quantize=quantize_resnet, get_controllers=controllers_resnet, network=ResNet, in_shape=(1,3,224,224), eps_in=_ILSVRC12_EPS, D=2**19, bs=53, get_in_shape=None, load_dataset_fn=load_ilsvrc12, transform=ILSVRC12PACTQuantTransform, quant_transform_args={'n_q':256}, n_levels_in=256, export_fn=export_net, code_size=160000),
    'ResNetCIFAR': QuantUtil(problem='CIFAR10', topo='ResNet', quantize=quantize_resnet_cifar, get_controllers=controllers_resnet_cifar, network=ResNetCIFAR, in_shape=(1,3,32,32), eps_in=_CIFAR10_EPS, D=2**19, bs=128, get_in_shape=None, load_dataset_fn=load_cifar10, transform=CIFAR10PACTQuantTransform, quant_transform_args={'n_q':256}, n_levels_in=256, export_fn=export_net, code_size=110000),
    'dvs_cnn' : QuantUtil(problem='DVS128', topo='dvs_cnn', quantize=quantize_dvsnet, get_controllers=controllers_dvsnet, network=DVSHybridNet, network_args={'inject_eps':False}, in_shape=None, eps_in=1., D=2**19, bs=128, get_in_shape=get_in_shape_dvsnet, load_dataset_fn=load_dvs128, transform=DVSAugmentTransform, n_levels_in=3, export_fn=export_dvsnet, code_size=340000),
    'simpleCNN': QuantUtil(problem='MNIST', topo='simpleCNN', quantize=quantize_simpleCNN, get_controllers=controllers_simpleCNN, network=simpleCNN, in_shape=(1,1,32,32), eps_in=_MNIST_EPS, D=2**19, bs=256, get_in_shape=None, load_dataset_fn=load_mnist, transform=MNISTPACTQuantTransform, quant_transform_args={'n_q':256}, n_levels_in=256, export_fn=export_net, code_size=150000)
}

# the topology directory where the specified network is defined
def get_topology_dir(key : str):
    topo = _QUANT_UTILS[key].topo
    return _QL_ROOTPATH.joinpath('systems').joinpath(get_system(key)).joinpath(topo)

# the QuantLab problem being solved by the specified network.
def get_system(key : str):
    return _QUANT_UTILS[key].problem

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
    if qu.in_shape is None:
        qu.in_shape = qu.get_in_shape(cfg)
        _QUANT_UTILS[key].in_shape = qu.in_shape

    net_cfg.update(qu.network_args)
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

def get_dataloader(key : str, cfg : dict, quantize : str, pad_img : Optional[int] = None, clip : bool = False):
    qu = _QUANT_UTILS[key]
    if torch.cuda.is_available():
        bs = torch.cuda.device_count() * qu.bs
    else:
        # network will be executed on CPU (not recommended!!)
        bs = 16
    ds = get_valid_dataset(key, cfg, quantize, pad_img=pad_img, clip=clip)
    return torch.utils.data.DataLoader(ds, bs)


def validate(net : nn.Module, dl : torch.utils.data.DataLoader, print_interval : int = 10, n_valid_batches : int = None):
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
        if ((i+1)%print_interval == 0):
            print(f'Accuracy after {i+1} batches: {n_correct/n_tot}')
        if (i+1) == n_valid_batches:
            break

    print(f'Final accuracy: {n_correct/n_tot}')
    net.to('cpu')


def get_input_channels(net : fx.GraphModule):
    for node in net.graph.nodes:
        if node.op == 'call_module' and isinstance(module_of_node(net, node), (nn.Conv1d, nn.Conv2d)):
            conv = module_of_node(net, node)
            return conv.in_channels

# THIS IS WHERE THE BUSINESS HAPPENS!
def integerize_network(net : nn.Module, key : str, fix_channels : bool, dory_harmonize : bool, word_align_channels : bool, requant_node : bool = False):
    qu = _QUANT_UTILS[key]
    # All we need to do to integerize a fake-quantized network is to run the
    # IntegerizePACTNetPass on it! Afterwards, the ONNX graph it produces will
    # contain only integer operations. Any divisions in the integerized graph
    # will be by powers of 2 and can be implemented as bit shifts.
    in_shp = qu.in_shape
    if key == 'dvs_cnn':
        in_shp_cnn = (in_shp[0], in_shp[1]//net.tcn_window, in_shp[2], in_shp[3])
        in_shp_tcn = (1, net.tcn.features[0].in_channels, net.tcn_window)
        tcn_eps_in = net.cnn.features[-1].get_eps()
        cnn_int_pass = IntegerizePACTNetPass(shape_in=in_shp_cnn, eps_in=qu.eps_in, D=qu.D, n_levels_in=qu.n_levels_in, fix_channel_numbers=fix_channels, word_align_channels=word_align_channels)
        cnn_int = cnn_int_pass(net.cnn)
        tcn_int_pass = IntegerizePACTNetPass(shape_in=in_shp_tcn, eps_in=tcn_eps_in, D=qu.D, n_levels_in=qu.n_levels_in, fix_channel_numbers=fix_channels)
        net.tcn.classifier = get_new_classifier(net.tcn.classifier)
        net.tcn.cls_replaced = True

        tcn_int = tcn_int_pass(net.tcn)
        if fix_channels:
            in_shp_l_cnn = list(in_shp_cnn)
            in_shp_l_cnn[1] = get_input_channels(cnn_int)
            in_shp_cnn = tuple(in_shp_l_cnn)
            in_shp_l_tcn = list(in_shp_tcn)
            in_shp_l_tcn[1] = get_input_channels(tcn_int)
            in_shp_tcn = tuple(in_shp_l_tcn)
        if dory_harmonize:
            dory_harmonize_pass_cnn = DORYHarmonizePass(in_shape=in_shp_cnn)
            cnn_int = dory_harmonize_pass_cnn(cnn_int)
            dory_harmonize_pass_tcn = DORYHarmonizePass(in_shape=in_shp_tcn)
            tcn_int = dory_harmonize_pass_tcn(tcn_int)
        return cnn_int, tcn_int
    else:
        int_pass = IntegerizePACTNetPass(shape_in=in_shp, eps_in=qu.eps_in, D=qu.D, n_levels_in=qu.n_levels_in, fix_channel_numbers=fix_channels, requant_node=requant_node)
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

def export_integerized_network(net : nn.Module, cfg : dict, key : str, export_dir : str, name : str, in_idx : int = 42, pad_img : Optional[int] = None, clip : bool = False, change_n_levels : int = None):
    qu = _QUANT_UTILS[key]
    # use a real image from the validation set
    ds = get_valid_dataset(key, cfg, quantize='int', pad_img=pad_img, clip=clip)
    test_input = ds[in_idx][0].unsqueeze(0)
    if key == 'dvs_cnn':
        qu.export_fn(*net, name=name, out_dir=export_dir, eps_in=qu.eps_in, integerize=False, D=qu.D, in_data=test_input, change_n_levels=change_n_levels, code_size=qu.code_size)
    else:
        qu.export_fn(net, name=name, out_dir=export_dir, eps_in=qu.eps_in, integerize=False, D=qu.D, in_data=test_input, code_size=qu.code_size)

def export_unquant_net(net : nn.Module, cfg : dict, key : str, export_dir : str, name : str):
    out_path = Path(export_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    onnx_file = f"{name}.onnx"
    onnx_path = out_path.joinpath(onnx_file)
    ds = get_valid_dataset(key, cfg, quantize='none')
    test_input = ds[42][0].unsqueeze(0)
    torch.onnx.export(net.to('cpu'),
                      test_input,
                      str(onnx_path),
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True)


# quite hacky but there is no other way

def get_new_classifier(classifier: PACTConv1d):
    new_classifier = nn.Sequential(nn.Flatten(),
                                   PACTLinear(
            in_features=classifier.in_channels*classifier.kernel_size[0],
            out_features=classifier.out_channels+1,
            bias=True,
            n_levels=classifier.n_levels,
            quantize=classifier.quantize,
            init_clip=classifier.init_clip,
            learn_clip=classifier.learn_clip,
            symm_wts=classifier.symm_wts,
            nb_std=classifier.nb_std,
            tqt=classifier.tqt,
            tqt_beta=classifier.tqt_beta,
            tqt_clip_grad=classifier.tqt_clip_grad))

    new_weights = classifier.weight.reshape(classifier.out_channels, -1)
    new_weights = torch.cat((new_weights, torch.zeros(new_weights.shape[1]).unsqueeze(0)))
    new_classifier[1].weight.data.copy_(new_weights)
    if classifier.bias is not None:
        new_classifier[1].bias.data.copy_(torch.cat((classifier.bias, torch.Tensor([0]))))
    else:
        new_classifier[1].bias.data.fill_(0)
    new_classifier[1].clip_lo = torch.nn.Parameter(torch.cat((classifier.clip_lo.squeeze(2), -torch.ones(1,1))))
    new_classifier[1].clip_hi = torch.nn.Parameter(torch.cat((classifier.clip_hi.squeeze(2), torch.ones(1,1))))
    new_classifier[1].clipping_params = classifier.clipping_params
    new_classifier[1].started = classifier.started
    return new_classifier



# if this script is executed directly, expose all the above functions as
# command line arguments.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QuantLab Example Flows')
    parser.add_argument('--net', required=True, type=str,
                        help='Network to treat - can be "MobileNetV1", "MobileNetV2", "MobileNetV3", "VGG" or "dvs_cnn"')
    parser.add_argument('--exp_id', required=True,
                        help='Experiment to integerize and export. The specified experiment must be a for a PACT/TQT-quantized network! Can be "best" or the index of the checkpoint')
    parser.add_argument('--ckpt_id', required=True, type=int,
                        help='Checkpoint to integerize and export. The specified checkpoint must be fully quantized with PACT/TQT!')
    parser.add_argument('--fix_channels', action='store_true',
                        help='Fix channels of conv layers for compatibility with DORY')
    parser.add_argument('--word_align_channels', action='store_true',
                        help='Fix channels of conv layers so (#input_ch * #input_bits) is a multiple of 32 to work around XpulpNN HW bug')
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
    parser.add_argument('--change_n_levels', type=int, default=None,
                        help='Only used in DVS128 export - override clipping bound of RequantShift modules of exported networks to this value')
    parser.add_argument('--code_size', type=int, default=None,
                        help="Override the default 'code reserved space' setting")
    parser.add_argument('--requant_node', action='store_true',
                        help='Export RequantShift nodes instead of mul-add-div sequences in ONNX graph')
    parser.add_argument('--n_valid_batch', type=int, default=None,
                        help='number of validation batches to run')
    


    args = parser.parse_args()

    if args.export_dir is not None:
        export_name = args.net if args.export_name is None else args.export_name

    if args.code_size is not None:
        _QUANT_UTILS[args.net].code_size = args.code_size
    exp_id = int(args.exp_id) if args.exp_id.isnumeric() else args.exp_id

    print(f'Loading network {args.net}, experiment {exp_id}, checkpoint {args.ckpt_id}')
    qnet = get_network(args.net, exp_id, args.ckpt_id, quantized=True)

    exp_cfg = get_config(args.net, exp_id)
    if args.validate_fq:
        print(f'Validating fake-quantized network {args.net} on dataset {get_system(args.net)}')
        dl = get_dataloader(args.net, exp_cfg, quantize='fake')
        validate(qnet, dl, args.accuracy_print_interval, n_valid_batches=args.n_valid_batch)

    print(f'Integerizing network {args.net}')

    int_net = integerize_network(qnet, args.net, args.fix_channels, not args.no_dory_harmonize, args.word_align_channels, args.requant_node)

    if args.fix_channels:
        pad_img = get_input_channels(int_net[0] if isinstance(int_net, tuple) else int_net)
    else:
        pad_img = None

    if args.validate_tq:
        dl = get_dataloader(args.net, exp_cfg, quantize='int', pad_img=pad_img)
        validate(int_net, dl, args.accuracy_print_interval, n_valid_batches=args.n_valid_batch)

    if args.export_dir is not None:
        print(f'Exporting integerized network {args.net} to directory {args.export_dir} under name {export_name}')
        export_integerized_network(int_net, exp_cfg, args.net, args.export_dir, export_name, pad_img=pad_img, clip=args.clip_inputs, change_n_levels=args.change_n_levels)
        if args.export_unquant:
            net_unq = get_network(args.net, exp_id, args.ckpt_id, quantized=False)
            export_unquant_net(net_unq, exp_cfg, args.net, args.export_dir, export_name)

