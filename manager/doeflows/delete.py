# 
# delete.py
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
import shutil

from .experimentaldesign import ExperimentalDesignLogger
import manager
import manager.logbook.logsmanager


def delete(args):

    logger = ExperimentalDesignLogger(args.problem, args.topology, args.exp_design)
    logger.load_register()

    while True:

        message  = manager.QUANTLAB_PREFIX + "Deleting an experimental design is an irreversible action.\n"
        message += manager.QUANTLAB_PREFIX + "Are you sure you want to delete experimental design \"{}\"? [yes/no] [ENTER]".format(args.exp_design)
        print(message)

        print(manager.QUANTLAB_PREFIX, end="")
        choice = input().lower()
        if choice not in set(['yes', 'no']):
            print(manager.QUANTLAB_PREFIX + "Invalid answer. Please respond with 'yes' or 'no'.")
        else:
            break

    if choice == 'yes':

        # delete the experimental units' logs folders
        for eu_status, eu_id, *eu_dofs_values in logger.eu_register:
            shutil.rmtree(os.path.join(logger.path_logs, manager.logbook.logsmanager._FORMAT_EXP_DIR.format(eu_id)))

        # delete the experimental design's log file
        logger.delete_edl()
