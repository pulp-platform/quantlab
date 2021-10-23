# 
# logsmanager.py
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

import math
import os
import shutil
import json
import torch
from torch.utils.tensorboard import SummaryWriter

from manager.meter import WriterStub
import manager.platform
import manager

from typing import Union, List


def _get_format_string(MAX):
    return "".join(["{:0", str(math.floor(math.log10(MAX))), "d}"])


_MAX_EXP_UNITS  = 10000
_FORMAT_EXP_DIR = "".join(["exp", _get_format_string(_MAX_EXP_UNITS)])

_MAX_CV_FOLDS    = 10
_FORMAT_FOLD_DIR = "".join(["fold", _get_format_string(_MAX_CV_FOLDS)])

_MAX_EPOCHS       = 1000
_FORMAT_CKPT_FILE = "".join(["epoch", _get_format_string(_MAX_EPOCHS), ".ckpt"])


class LogsManager(object):

    def __init__(self, path_logs: str, exp_id: int):
        """The entity that manages logging interactions with the disk.

        This object stores to and loads from disk all the files that should be
        non-volatile. Examples include JSON configuration files, PyTorch
        checkpoint files, TensorBoard event files.

        Args:
            path_logs: the path to the current machine learning system's logs
                folder.
            exp_id: the current experimental unit's identifier.

        Attributes:
            path_exp (str): the path to the current experimental unit's logs
                folder.
            _fold_id (int): the identifier of the fold being trained, or from
                which a checkpoint is being tested.
            _path_fold (str): the path to the current fold's logs folder.
            _path_saves (str): the path to the current fold's logs sub-folder
                dedicated to PyTorch checkpoints.
            _path_stats (str): the path to the current fold's logs sub-folder
                dedicated to TensorBoard event files.
            _writerstub_epoch (:py:class:`manager.meter.WriterStub`): the stub
                wrapping the ``SummaryWriter``s that log statistics once per
                epoch.
            _writerstub_step_train (:py:class:`manager.meter.WriterStub`): the
                stub wrapping the ``SummaryWriter``s that log statistics at
                each training step.
            _writerstub_step_valid (:py:class:`manager.meter.WriterStub`): the
                stub wrapping the ``SummaryWriter``s that log statistics at
                each validations step.
        """

        message = manager.QUANTLAB_PREFIX + "Experimental unit #{}.".format(exp_id)
        print(message)

        # get path to the experimental unit's log folder
        self.path_exp = os.path.join(path_logs, _FORMAT_EXP_DIR.format(exp_id))

        # cross-validation fold handlers
        self._fold_id    = None
        self._path_fold  = None
        self._path_saves = None
        self._path_stats = None

        self._writerstub_epoch      = WriterStub()
        self._writerstub_step_train = WriterStub()
        self._writerstub_step_valid = WriterStub()

    # === EXPERIMENTAL UNIT'S LOGS FOLDER FUNCTIONALITIES ===
    def create_exp_folder(self) -> None:

        os.mkdir(self.path_exp)

        message = manager.QUANTLAB_PREFIX + "Experimental unit's logs folder created at <{}>.".format(self.path_exp)
        print(message)

    def check_exp_folder(self) -> None:

        assert os.path.isdir(self.path_exp)

        message = manager.QUANTLAB_PREFIX + "Experimental unit's logs folder at <{}>.".format(self.path_exp)
        print(message)

    def destroy_exp_folder(self) -> None:

        message  = manager.QUANTLAB_PREFIX + "Deleting an experimental unit's logs folder is an irreversible action.\n"
        message += manager.QUANTLAB_PREFIX + "Are you sure you want to delete experimental unit #{}? [yes/no] [ENTER]".format(int(os.path.basename(self.path_exp).replace('exp', '')))
        print(message)

        while True:

            print(manager.QUANTLAB_PREFIX, end="")

            choice = input().lower()
            if choice not in set(['yes', 'no']):
                print(manager.QUANTLAB_PREFIX + "Invalid answer. Please respond with 'yes' or 'no'.")
            else:
                break

        if choice == 'yes':
            shutil.rmtree(self.path_exp)

    # === CONFIGURATION FILE FUNCTIONALITIES ===
    def store_config(self, config: dict) -> None:
        """Store a configuration dictionary to a JSON file.

        Args:
            config: the functional description of an experimental unit; the
                logbook will exchange this information with the assistants to
                assemble the required machine learning system.
        """

        config_file = os.path.join(self.path_exp, 'config.json')
        with open(config_file, 'w') as fp:
            json.dump(config, fp, indent=4)

    def load_config(self) -> dict:
        """Load a JSON file into a configuration dictionary.

        Returns:
            config: the functional description of an experimental unit; the
                logbook will exchange this information with the assistants to
                assemble the required machine learning system.
        """

        config_file = os.path.join(self.path_exp, 'config.json')
        with open(config_file, 'r') as fp:
            config = json.load(fp)

        return config

    # === FOLD IDENTIFICATION FUNCTIONALITIES ===
    @property
    def fold_id(self) -> int:
        return self._fold_id

    def set_fold_id(self, fold_id: int) -> None:
        self._fold_id = fold_id

    def discover_fold_id(self) -> None:
        """Discover the most recently updated cross-validation fold.

        QuantLab assumes that each experimental unit is a cross-validation
        experimental unit, consisting of :math:`K \geq 1` folds. Since the
        concept of cross-validation is meant to be applied at training time,
        QuantLab also assumes that fold logs folders will only be created at
        training time. I.e., for a given experimental unit, no other flow
        apart from the training flow should create new fold logs folders.
        Finally, QuantLab assumes that the folds will be created sequentially.
        In fact, QuantLab does not support parallelism at the fold level (it
        can only speed up each fold's training, or the test of a specific
        checkpoint). Hence, I assume that since the :math:`\bar{k}`-th fold
        can be created only after all the folds
        :math:`k = 0, \dots, \bar{k}-1` have been completed, the largest index
        identifies the most recently created (and possibly not run to
        completion) fold.
        """

        try:
            self._fold_id = max([int(f.replace('fold', '')) for f in os.listdir(self.path_exp) if f.startswith('fold')])
        except ValueError:  # the fact that no fold is found should mean that a new experiment is starting
            self._fold_id = 0

    # === STATISTICS LOGGING FUNCTIONALITIES ===
    @property
    def _path_stats_epoch(self):
        return os.path.join(self._path_stats, 'epoch')

    @property
    def _path_stats_step_train(self):
        return os.path.join(self._path_stats, 'step', 'train')

    @property
    def _path_stats_step_valid(self):
        return os.path.join(self._path_stats, 'step', 'valid')

    def setup_fold_logs(self, fold_id: int) -> None:
        """Get the pointers to a fold's logs folder and sub-folders.

        To structure the logging of each experimental run, QuantLab separates
        log files by type:
            * the ``saves`` sub-folder hosts PyTorch checkpoint files;
            * the ``stats`` sub-folder hosts TensorBoard event files.

        Args:
            fold_id: the identifier of the cross-validation fold to which the
                :obj:`LogsManager` should point.
        """

        self._fold_id = fold_id
        self._path_fold = os.path.join(self.path_exp, _FORMAT_FOLD_DIR.format(self._fold_id))

        # create folder to store checkpoints (i.e., PyTorch checkpoint files)
        self._path_saves = os.path.join(self._path_fold, 'saves')
        os.makedirs(self._path_saves, exist_ok=True)

        # create folder to log statistics (i.e., TensorBoard event files)
        self._path_stats = os.path.join(self._path_fold, 'stats')
        os.makedirs(self._path_stats_epoch, exist_ok=True)
        os.makedirs(self._path_stats_step_train, exist_ok=True)
        os.makedirs(self._path_stats_step_valid, exist_ok=True)

    @property
    def writerstub_epoch(self) -> WriterStub:
        return self._writerstub_epoch

    @property
    def writerstub_step_train(self) -> WriterStub:
        return self._writerstub_step_train

    @property
    def writerstub_step_valid(self) -> WriterStub:
        return self._writerstub_step_valid

    def create_writers(self, start_epoch_id: int, n_batches_train: int, n_batches_valid: int) -> None:
        self._writerstub_epoch.writer      = SummaryWriter(log_dir=self._path_stats_epoch,      purge_step=start_epoch_id)
        self._writerstub_step_train.writer = SummaryWriter(log_dir=self._path_stats_step_train, purge_step=start_epoch_id * n_batches_train)
        self._writerstub_step_valid.writer = SummaryWriter(log_dir=self._path_stats_step_valid, purge_step=start_epoch_id * n_batches_valid)

    def destroy_writers(self) -> None:
        self._writerstub_epoch.writer.close()
        self._writerstub_step_train.writer.close()
        self._writerstub_step_valid.writer.close()

    # === CHECKPOINTING FUNCTIONALITIES ===
    def make_ckpt_path(self, ckpt_name: str) -> str:
        return os.path.join(self._path_saves, ckpt_name)

    def store_checkpoint(self,
                         epoch_id: int,
                         net: torch.nn.Module,
                         opt: torch.optim.Optimizer,
                         lr_sched: Union[torch.optim.lr_scheduler._LRScheduler, None],
                         qnt_ctrls: List[None],
                         meter_train,
                         meter_valid,
                         is_best: bool = False) -> None:

        ckpt = {}

        ckpt['experiment']             = {}
        ckpt['experiment']['fold_id']  = self._fold_id
        ckpt['experiment']['epoch_id'] = epoch_id

        ckpt['net']                    = net.state_dict()

        ckpt['gd']                     = {}
        ckpt['gd']['opt']              = opt.state_dict()
        ckpt['gd']['lr_sched']         = lr_sched.state_dict() if lr_sched is not None else None

        ckpt['qnt_ctrls']              = [qc.state_dict() for qc in qnt_ctrls]

        ckpt['train_meter'] = {}
        ckpt['train_meter']['best_loss']  = meter_train.best_loss
        ckpt['valid_meter'] = {}
        ckpt['valid_meter']['best_loss'] = meter_valid.best_loss

        ckpt_filename = self.make_ckpt_path('best.ckpt' if is_best else _FORMAT_CKPT_FILE.format(epoch_id))
        torch.save(ckpt, ckpt_filename)

        message = manager.QUANTLAB_PREFIX + "Checkpoint stored: <{}>.".format(ckpt_filename)
        print(message)

    def load_checkpoint(self,
                        platform: manager.platform.PlatformManager,
                        net: torch.nn.Module,
                        opt: torch.optim.Optimizer,
                        lr_sched: Union[torch.optim.lr_scheduler._LRScheduler, None],
                        qnt_ctrls: List[None],
                        train_meter: Union[object, None] = None,  # `object` is of type `manager.meter.Meter`
                        valid_meter: Union[object, None] = None,  # `object` is of type `manager.meter.Meter`
                        ckpt_id: Union[int, None] = None) -> int:

        ckpts_list = os.listdir(self._path_saves)

        if len(ckpts_list) > 0:  # a checkpoint exists

            if ckpt_id is None:  # discover most recent checkpoint; this mode is used when resuming crashed/interrupted experimental runs
                ckpt_filename = max([self.make_ckpt_path(f) for f in ckpts_list], key=os.path.getctime)
            else:  # load requested checkpoint; this mode should only be used by the test flow
                ckpt_filename = self.make_ckpt_path('best.ckpt' if ckpt_id == -1 else _FORMAT_CKPT_FILE.format(ckpt_id))

            # load the checkpoint into the proper structures
            ckpt = torch.load(ckpt_filename)
            assert ckpt['experiment']['fold_id'] == self._fold_id  # if not, something weird must have happened during a call to `store_checkpoint`...

            # epoch ID
            epoch_id = ckpt['experiment']['epoch_id']

            # network parameters
            is_nndataparallel_state_dict = all(k.startswith('module.') for k in ckpt['net'].keys())
            if platform.is_nndataparallel_run != is_nndataparallel_state_dict:
                if not platform.is_nndataparallel_run:
                    assert is_nndataparallel_state_dict
                    ckpt['net'] = {"module.{}".format(k): v for k, v in ckpt['net'].items()}
                else:
                    assert platform.is_nndataparallel_run and (not is_nndataparallel_state_dict)
                    ckpt['net'] = {k.replace("module."): v for k, v in ckpt['net'].items()}
            net.load_state_dict(ckpt['net'])

            # optimizer hyper-parameters
            opt.load_state_dict(ckpt['gd']['opt'])
            if lr_sched is not None:
                assert ckpt['gd']['lr_sched'] is not None
                lr_sched.load_state_dict(ckpt['gd']['lr_sched'])
            for c, sd in zip(qnt_ctrls, ckpt['qnt_ctrls']):
                c.load_state_dict(sd)

            # meter statistics
            if train_meter is not None:
                train_meter.best_loss = ckpt['train_meter']['best_loss']
            if valid_meter is not None:
                valid_meter.best_loss = ckpt['valid_meter']['best_loss']

            message = manager.QUANTLAB_PREFIX + "Checkpoint found: <{}>.".format(ckpt_filename)

        else:  # we start from the first epoch, and the state of the objects will remain the one set by the initial conditions
            epoch_id = -1
            message = manager.QUANTLAB_PREFIX + "No checkpoint found at <{}>.".format(self._path_saves)

        print(message)

        return epoch_id

