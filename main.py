# 
# main.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
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

"""The facade of QuantLab.

Run this script to open the doors to all QuantLab flows.
"""

import argparse

from manager.flows import platform
from manager.flows import configure
from manager.flows import delete
from manager.flows import train
from manager.flows import test
from manager.flows import quantize


def parse_cli():

    # Command Line Interface
    parser = argparse.ArgumentParser(description="QuantLab")
    parser.add_argument('--problem',  required=True, type=str,                                                       help="Data set")
    parser.add_argument('--topology', required=True, type=str,                                                       help="Network topology")

    subparsers = parser.add_subparsers(dest="flow",                                                                  help="QuantLab flows")
    # platform inspection flow
    parser_platform = subparsers.add_parser('platform',                                                              help="Show computing system configuration (HW/OS)")
    parser_platform.add_argument('--horovod', required=False, action='store_true',                                   help="Distributed computing with Horovod")
    parser_platform.set_defaults(func=platform)
    # configuration flow
    parser_cfg = subparsers.add_parser('configure',                                                                  help="Configure an experimental unit")
    parser_cfg.add_argument('--target_loss',  required=False, type=str, default='valid', choices=('train', 'valid'), help="Whether to optimise training or validation loss")
    parser_cfg.add_argument('--ckpt_period',  required=False, type=int, default=5,                                   help="Checkpointing period (in epochs; default: 5 epochs)")
    parser_cfg.add_argument('--n_folds',      required=False, type=int, default=1,                                   help="Number of cross-validation folds (default: 1 fold)")
    parser_cfg.add_argument('--cv_seed',      required=False, type=int, default=-1,                                  help="The seed used by the algorithm that splits training data into folds")
    parser_cfg.add_argument('--fix_sampler',  required=False, action='store_true',                                   help="Use the same seed for the sampler(s) across different folds")
    parser_cfg.add_argument('--sampler_seed', required=False, type=int, default=-1,                                  help="The seed used by PyTorch sampler(s)")
    parser_cfg.add_argument('--fix_network',  required=False, action='store_true',                                   help="Use the same seed for the network initialisation across different folds")
    parser_cfg.add_argument('--network_seed', required=False, type=int, default=-1,                                  help="The seed used by the algorithm that initialises the network's parameters")
    parser_cfg.set_defaults(horovod=False)
    parser_cfg.set_defaults(func=configure)
    # delete flow
    parser_dlt = subparsers.add_parser('delete',                                                                     help="Delete an experimental unit")
    parser_dlt.add_argument('--exp_id',  required=True,  type=int,                                                   help="The ID of the experimental run to delete")
    parser_dlt.set_defaults(horovod=False)
    parser_dlt.set_defaults(func=delete)
    # training flow
    parser_train = subparsers.add_parser('train',                                                                    help="Launch a training or fine-tuning experimental run")
    parser_train.add_argument('--exp_id',  required=True,  type=int,                                                 help="The ID of the experimental run to launch or resume")
    parser_train.add_argument('--horovod', required=False, action='store_true',                                      help="Distributed data parallel training with Horovod")
    parser_train.set_defaults(func=train)
    # testing flow
    parser_test = subparsers.add_parser('test',                                                                      help="Assess the performance of a trained model on test data")
    parser_test.add_argument('--exp_id',  required=True,  type=int,                                                  help="The ID of the experimental unit to which the trained model belongs")
    parser_test.add_argument('--fold_id', required=True,  type=int,                                                  help="The ID of the cross-validation fold to which the trained model belongs")
    parser_test.add_argument('--ckpt_id', required=False, type=int, default=-1,                                      help="The trained model checkpoint's file ID (default: best model)")
    parser_test.add_argument('--horovod', required=False, action='store_true',                                       help="Distributed data parallel test with Horovod")
    parser_test.set_defaults(func=test)
    # quantization flow
    parser_qnt = subparsers.add_parser('quantize',                                                                   help="Launch interactive graph editing session")
    parser_qnt.add_argument('--exp_id',  required=True,  type=int,                                                   help="The ID of the experimental unit to which the model belongs")
    parser_qnt.add_argument('--fold_id', required=False, type=int,                                                   help="The ID of the cross-validation fold to which the trained model belongs")
    parser_qnt.add_argument('--ckpt_id', required=False, type=int, default=-1,                                       help="The trained model checkpoint's file ID (default: best model)")
    parser_qnt.set_defaults(horovod=False)
    parser_qnt.set_defaults(func=quantize)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_cli()
    args.func(args)

