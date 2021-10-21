#
# doemain.py
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

import argparse

from manager.doeflows import configure
from manager.doeflows import archive
from manager.doeflows import delete


def parse_cli():

    parser = argparse.ArgumentParser(description="QuantLab DoE")
    parser.add_argument('--problem',  required=True, type=str, help="Data set")
    parser.add_argument('--topology', required=True, type=str, help="Network topology")

    subparsers = parser.add_subparsers(dest='doeflows',                                                              help="QuantLab DoE flows")

    parser_cfg = subparsers.add_parser('configure',                                                                  help="Configure an experimental design")
    parser_cfg.add_argument('--exp_design',   required=True,  type=str,                                              help="The experimental design to configure")
    parser_cfg.add_argument('--target_loss',  required=False, type=str, default='valid', choices=('train', 'valid'), help="Whether to optimise training or validation loss")
    parser_cfg.add_argument('--ckpt_period',  required=False, type=int, default=5,                                   help="Checkpointing period (in epochs; default: 5 epochs)")
    parser_cfg.add_argument('--n_folds',      required=False, type=int, default=1,                                   help="Number of cross-validation folds (default: 1 fold)")
    parser_cfg.add_argument('--cv_seed',      required=False, type=int, default=-1,                                  help="The seed used by the algorithm that splits training data into folds")
    parser_cfg.add_argument('--fix_sampler',  required=False, action='store_true',                                   help="Use the same seed for the sampler(s) across different folds")
    parser_cfg.add_argument('--sampler_seed', required=False, type=int, default=-1,                                  help="The seed used by PyTorch sampler(s)")
    parser_cfg.add_argument('--fix_network',  required=False, action='store_true',                                   help="Use the same seed for the network initialisation across different folds")
    parser_cfg.add_argument('--network_seed', required=False, type=int, default=-1,                                  help="The seed used by the algorithm that initialises the network's parameters")
    parser_cfg.set_defaults(func=configure)

    parser_arx = subparsers.add_parser('archive',                                                                    help="Archive all the experimental units belonging to an experimental design")
    parser_arx.add_argument('--exp_design',   required=True,  type=str,                                              help="The experimental design to archive")
    parser_arx.add_argument('--save_storage', required=False, action='store_true',                                   help="Delete experimental units' logs folders as soon as they have been archived")
    parser_arx.set_defaults(func=archive)

    parser_dlt = subparsers.add_parser('delete',                                                                     help="Delete all the experimental units belonging to an experimental design")
    parser_dlt.add_argument('--exp_design',   required=True,  type=str,                                              help="The experimental design to delete")
    parser_dlt.set_defaults(func=delete)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_cli()
    args.func(args)
