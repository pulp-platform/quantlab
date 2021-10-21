# 
# configure.py
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

import os
import json
import subprocess
import io

from .experimentaldesign import ExperimentalUnitStatus, ExperimentalDesignLogger


import IPython


def get_experimental_unit_id(stdout):
    # The experimental unit's ID can be inferred from the standard output of
    # the process running the basic QuantLab flow `configure`; see:
    #
    #     manager/logbook/logsmanager.py:L84
    #
    stdout_str        = io.StringIO(stdout.decode('utf-8'))
    stdout_first_line = stdout_str.readline().rstrip()  # strip newline
    stdout_str_eu_id  = stdout_first_line.replace('[QuantLab] Experimental unit #', '').replace('.', '')

    return int(stdout_str_eu_id)


def build_command(args):

    base_command = 'python main.py --problem={} --topology={} configure'.format(args.problem, args.topology)
    target_loss  = '--target_loss={}'.format(args.target_loss)
    ckpt_period  = '--ckpt_period={}'.format(args.ckpt_period)
    n_folds      = '--n_folds={}'.format(args.n_folds)
    cv_seeds     = '--cv_seed={}'.format(args.cv_seed)
    if args.fix_sampler:
        sampler_seeding = '--fix_sampler --sampler_seed={}'.format(args.sampler_seed)
    else:
        sampler_seeding = ''
    if args.fix_network:
        network_seeding = '--fix_network --network_seed={}'.format(args.network_seed)
    else:
        network_seeding = ''

    return ' '.join([base_command, target_loss, ckpt_period, n_folds, cv_seeds, sampler_seeding, network_seeding])


def configure(args):
    """The purposes of the `ExperimentalDesignLogger` are:

    * creating valid QuantLab configuration dictionaries;
    * registering configured experiments on a log file that is specific to the
      experimental design.

    In-between these steps, the basic QuantLab flow `configure` needs to be
    called. This is not the purpose of the logger but of the `configure` flow
    itself.

    """

    logger = ExperimentalDesignLogger(args.problem, args.topology, args.exp_design)

    logger.write_args(args)
    logger.write_dofs()

    logger.load_register()
    start_idx = len(logger.eu_register)  # TODO

    for idx, setup in enumerate(logger.ed.setups):

        if idx < start_idx:  # some experimental units have already been configured # TODO
            continue                                                                # TODO

        # patch the configuration
        config = logger.ed.patch_base_config(setup)

        # Write the patched config to the system package, since the basic
        # QuantLab flow `configure` will fetch it from that folder.
        with open(os.path.join('systems', args.problem, args.topology, 'config.json'), 'w') as fp:
            json.dump(config, fp, indent=4)

        # call the basic QuantLab flow `configure`
        command = build_command(args)
        output  = subprocess.run(command, shell=True, capture_output=True)
        eu_id   = get_experimental_unit_id(output.stdout)

        # update the record of configured experimental units
        logger.update_register(ExperimentalUnitStatus.CONFIGURED, eu_id, setup)
        logger.write_register()
