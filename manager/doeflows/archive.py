# 
# archive.py
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
import subprocess
import shutil

from .experimentaldesign import ExperimentalDesignLogger
import manager
import manager.logbook.logsmanager

from typing import Union


def archive_exp(path_archive: Union[str, os.PathLike], eu_id: int, save_storage: bool) -> None:

    dir_logs = os.path.dirname(path_archive)
    dir_exp = manager.logbook.logsmanager._FORMAT_EXP_DIR.format(eu_id)
    path_exp = os.path.join(dir_logs, dir_exp)
    file_tar = '.'.join([dir_exp, 'tar', 'gz'])
    path_tar = os.path.join(path_archive, file_tar)

    subprocess.run('tar -pczf {} -C {} {}'.format(path_tar, os.path.dirname(path_exp), os.path.basename(path_exp)), shell=True)

    if save_storage:
        shutil.rmtree(path_exp)


def archive(args):

    logger = ExperimentalDesignLogger(args.problem, args.topology, args.exp_design)
    logger.load_register()

    if logger.is_experiment_archivable():

        path_archive = logger.get_path_archive()

        for eu_status, eu_id, *eu_dofs_values in logger.eu_register:
            archive_exp(path_archive, eu_id, args.save_storage)

        logger.move_edl(path_archive)
        if args.save_storage:
            logger.delete_edl()

    else:
        print(manager.QUANTLAB_PREFIX + "Some experiments have not been executed to completion. Aborting archiviation flow.")
