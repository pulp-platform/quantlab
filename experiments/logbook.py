import sys
import os
import shutil

from collections import OrderedDict

import importlib
import json
import math
import random
from tensorboardX import SummaryWriter
import torch


__MAX_EXPERIMENTS__ = 1000
__EXP_ALIGN__ = math.ceil(math.log10(__MAX_EXPERIMENTS__))  # experiment ID string length (decimal literal)
__MAX_SEED__ = 10000
__MAX_CV_FOLDS__ = 10
__CV_FOLDS_ALIGN__ = math.ceil(math.log10(__MAX_CV_FOLDS__))  # cross-validation fold ID string length (decimal literal)
__MAX_EPOCHS__ = 10000
__EPOCHS_ALIGN__ = math.ceil(math.log10(__MAX_EPOCHS__))  # checkpoint ID string length (decimal literal)


class Logbook(object):

    def __init__(self, problem, topology, exp_id, verbose=True):
        """Experiment management abstraction.

        The logbook registers the information needed by the *lab daemons* to
        instantiate individual and treatment, plus the status of the current
        experiment.

        Args:
            problem (str): The data set name.
            topology (str): The network topology used to solve the data set.
            exp_id (str): The decimal literal identifying the experiment.
            verbose (bool): Whether status messages should be printed.

        Attributes:
            problem (str): The data set name.
            topology (str): The network topology used to solve the data set.
            lib (module): The Python module implementing the network topology.
                Besides the network definition itself, the module can include
                network-specific data pre- and post-processing routines,
                network-specific loss functions/optimizers/learning rate
                schedulers, and network-specific performance metrics (and the
                respective meters). Moreover, the module can include scripts
                that define how to convert the network to a quantized version
                and/or which statistics will be tracked during training (e.g.,
                weights, activations, gradients, quantization parameters).
            dir_data (str): The full path to the data set folder.
                This path is actually a symbolic link to the real data folder,
                which should have been created on a fast storage device (e.g.,
                a SSD drive) to attain better training speed.
            dir_log (str): The full path to the log folder.
                This path is actually a symbolic link to the real log folder,
                which should have been created on a high-capacity storage
                device (e.g., a HDD) to store a large number of PyTorch
                checkpoint files and TensorBoard event files.
            dir_save (str): The full path to the checkpoints folder.
                PyTorch checkpoints about the current fold will be stored
                here.
            dir_stats (str): The full path to the statistics folder.
                TensorBoard events about the current fold will be stored here.
            writer (:obj:`SummaryWriter`): The object that logs the experiment
                configuration and statistics on a TensorBoard event file.
            config (:obj:`dict`): The full description of the experiment.
            ckpt (None or :obj:`dict`): The selected checkpoint from which the
                experiment should be resumed.
            best_metric (float): The best result achieved by the experiment up
                to the current iteration.
            i_epoch (int): The current iteration of the experiment.
            meter (:obj:`Meter`): The object which measures loss and data set-
                specific metrics.
            verbose (bool): Whether status messages should be printed.

        """
        self.verbose       = verbose
        self.problem       = problem
        self.topology      = topology
        self.lib           = importlib.import_module('.'.join(['problems', self.problem, self.topology]))
        self.dir_data      = None
        self.dir_logs      = None
        self.dir_exp       = None
        self._set_exp_folders(exp_id)
        self.config        = None
        self._load_exp_config()
        self.seed          = None
        self._cv_status    = None
        self.i_fold        = None
        self._load_cv_status()

        self.dir_saves     = None
        self.dir_stats     = None
        self.writer        = None
        self.ckpt          = None
        self.i_epoch       = None

        self.meter         = None
        self.metrics       = {
            'train_loss':   None,
            'train_metric': None,
            'valid_loss':   None,
            'valid_metric': None
        }
        self.target_metric = None
        # self.period_metric = None
        # self.track_metric  = None

    def _set_exp_folders(self, exp_id):
        """Get pointers to the data and experiment folders.

        Args:
            exp_id (str): The decimal literal identifying the experiment.
            fold_id (str): The decimal literal identifying the fold (for CV).

        """
        QUANT_HOME = sys.path[0]
        # get pointers to HARD SHARED resources
        HARD_STORAGE_JSON = os.path.join(QUANT_HOME, 'cfg', 'hard_storage.json')
        with open(HARD_STORAGE_JSON, 'r') as fp:
            d = json.load(fp)
            # data
            HARD_STORAGE_DATA_HOME = os.path.join(d['data'], 'Quant')
            HARD_DIR_DATA = os.path.join(HARD_STORAGE_DATA_HOME, 'problems', self.problem, 'data')
            if not os.path.isdir(HARD_DIR_DATA):
                raise FileNotFoundError('{} hard directory (data) not found: {}'.format(self.problem, HARD_DIR_DATA))
            # log
            HARD_STORAGE_LOGS_HOME = os.path.join(d['logs'], 'Quant')
            HARD_DIR_LOGS = os.path.join(HARD_STORAGE_LOGS_HOME, 'problems', self.problem, 'logs')
            if not os.path.isdir(HARD_DIR_LOGS):
                raise FileNotFoundError('{} hard directory (logs) not found: {}'.format(self.problem, HARD_DIR_LOGS))
        # get pointers to SOFT SHARED resources (which are redirected to HARD ones using symlinks)
        HOME_PROBLEM = os.path.join(QUANT_HOME, 'problems', self.problem)
        dir_data = os.path.join(HOME_PROBLEM, 'data')
        if not os.path.isdir(dir_data):
            os.symlink(HARD_DIR_DATA, dir_data)
        dir_logs = os.path.join(HOME_PROBLEM, 'logs')
        if not os.path.isdir(dir_logs):
            os.symlink(HARD_DIR_LOGS, dir_logs)
        # get pointers to PRIVATE experiment resources
        if exp_id:
            # retrieve an existing report
            exp_id = int(exp_id)
        else:
            # create a new report
            exp_folders = [f for f in os.listdir(dir_logs) if f.startswith('exp')]
            if len(exp_folders) == 0:
                exp_id = 0
            else:
                exp_id = max(int(f.replace('exp', '')) for f in exp_folders) + 1
        dir_exp = os.path.join(dir_logs, 'exp'+str(exp_id).rjust(__EXP_ALIGN__, '0'))
        if not os.path.isdir(dir_exp):
            os.mkdir(dir_exp)
        self.dir_data = dir_data
        self.dir_logs = dir_logs
        self.dir_exp  = dir_exp
        if self.verbose:
            # print setup message
            message  = 'EXPERIMENT LOGBOOK\n'
            message += 'Problem:               {}\n'.format(self.problem)
            message += 'Network topology:      {}\n'.format(self.topology)
            message += 'Data directory:        {}\n'.format(self.dir_data)
            message += 'Logs directory:        {}\n'.format(self.dir_logs)
            message += 'Experiment directory:  {}\n'.format(self.dir_exp)

            def print_message(message):
                """Print a nice delimiter around a multiline message."""
                lines = message.splitlines()
                tab_size = 4
                width = max(len(l) for l in lines) + tab_size
                print('+' + '-' * width + '+')
                for l in lines:
                    print(l)
                print('+' + '-' * width + '+')

            print_message(message)

    def _load_exp_config(self):
        private_config_file = os.path.join(self.dir_exp, 'config.json')
        if not os.path.isfile(private_config_file):
            # no configuration in the experiment folder: look for global one
            shared_config_file = os.path.join(os.path.dirname(self.lib.__file__), 'config.json')
            if not os.path.isfile(shared_config_file):
                raise FileNotFoundError('Configuration file not found: {}'.format(shared_config_file))
            shutil.copyfile(shared_config_file, private_config_file)
        with open(private_config_file) as fp:
            self.config = json.load(fp)

    def _fold_folder_name(self, i_fold):
        return 'fold'+str(i_fold).rjust(__CV_FOLDS_ALIGN__, '0')

    def _store_cv_status(self, cv_file, seed, cv_status):
        d = OrderedDict({'seed': seed})
        for i_fold in range(self.config['experiment']['n_folds']):
            d.update({self._fold_folder_name(i_fold): cv_status[i_fold]})
        with open(cv_file, 'w') as fp:
            json.dump(d, fp, indent=4)

    def _load_cv_status(self):
        cv_file = os.path.join(self.dir_exp, 'cv.json')
        if not os.path.isdir(cv_file):
            seed = random.randint(0, __MAX_SEED__)
            cv_status = [0 for i_fold in range(0, self.config['experiment']['n_folds'])]
            self._store_cv_status(cv_file, seed, cv_status)
            i_fold = 0
        else:
            with open(cv_file, 'r') as fp:
                d = OrderedDict(json.load(fp))
            seed = d['seed']
            cv_status = list([v for k, v in d.items() if k != 'seed'])
            try:
                i_fold = cv_status.index(0)
                # folds must have been processed (and completed) in sequence
                assert not any(cv_status[i_fold+1:]), 'The CV experiment is corrupted! See {} for details.'.format(cv_file)
            except ValueError:
                i_fold = self.config['experiment']['n_folds']
        self.seed       = 1000
        self._cv_status = cv_status
        self.i_fold     = i_fold
        if self.verbose:
            print('[{}/{}] fold(s) completed'.format(self.i_fold, self.config['experiment']['n_folds']))

    def _set_fold_folders(self):
        """Find the first incomplete fold and load most recent checkpoint."""
        # find fold directory
        dir_fold = os.path.join(self.dir_exp, self._fold_folder_name(self.i_fold))
        if not os.path.isdir(dir_fold):
            os.mkdir(dir_fold)
        dir_saves = os.path.join(dir_fold, 'saves')
        if not os.path.isdir(dir_saves):
            os.mkdir(dir_saves)
        dir_stats = os.path.join(dir_fold, 'stats')
        if not os.path.isdir(dir_stats):
            os.mkdir(dir_stats)
        # set pointers
        self.dir_saves = dir_saves
        self.dir_stats = dir_stats
        # set TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.dir_stats)

    def load_checkpoint(self, load):
        ckpt_list = os.listdir(self.dir_saves)
        if len(ckpt_list) == 0:
            # no checkpoints were found
            ckpt = None
            i_epoch = 0
        else:
            # some checkpoints were found
            if load == 'best':
                ckpt_file = os.path.join(self.dir_saves, 'best.ckpt')
            elif load == 'last':
                ckpt_file = max([os.path.join(self.dir_saves, f) for f in ckpt_list], key=os.path.getctime)
            # else:
            #     ckpt_id = str(load).rjust(__EPOCHS_ALIGN__, '0')
            #     ckpt_name = 'epoch' + ckpt_id + '.ckpt'
            #     ckpt_file = os.path.join(self.dir_saves, ckpt_name)
            if self.verbose:
                print('Loading checkpoint {} ...'.format(ckpt_file))
            ckpt = torch.load(ckpt_file)
            if self.verbose:
                print('\b done!')
            i_epoch = ckpt['fold']['i_epoch'] + 1  # start new epoch
        self.ckpt = ckpt
        self.i_epoch = i_epoch

    def store_checkpoint(self, ckpt, is_best=False):
        """Store states of network and training procedure."""
        if is_best:
            ckpt_name = 'best.ckpt'
        else:
            ckpt_id   = str(ckpt['fold']['i_epoch']).rjust(__EPOCHS_ALIGN__, '0')
            ckpt_name = 'epoch'+ckpt_id+'.ckpt'
        ckpt_file = os.path.join(self.dir_saves, ckpt_name)
        if self.verbose:
            print('Storing checkpoint {} ...'.format(ckpt_file))
        torch.save(ckpt, ckpt_file)
        if self.verbose:
            print('\b done!')
        self.ckpt = ckpt

    def init_measurements(self):
        meter_module = importlib.import_module('.'.join(['problems', self.problem, 'meter']))
        self.meter   = getattr(meter_module, 'Meter')(self.lib.postprocess_pr, self.lib.postprocess_gt)
        if self.ckpt is None:
            self.metrics = {
                'train_loss': math.inf,
                'train_metric': self.meter.start_metric,
                'valid_loss': math.inf,
                'valid_metric': self.meter.start_metric
            }
        else:
            self.metrics.update(self.ckpt['fold']['metrics'])
        self.target_metric = self.config['experiment']['metrics']['update_metric']
        # self.period_metric = self.config['experiment']['metrics']['period_metric']

    def is_better(self, stats):
        if self.target_metric.endswith('loss'):
            # loss has decreased
            is_better = stats[self.target_metric] < self.metrics[self.target_metric]
        else:
            # problem main metric has improved
            is_better = self.meter.is_better(stats[self.target_metric], self.metrics[self.target_metric])
        if is_better:
            self.metrics.update(stats)
        return is_better

    def open_fold(self):
        self._set_fold_folders()

    def close_fold(self):
        self._cv_status[self.i_fold] = 1
        self._store_cv_status(os.path.join(self.dir_exp, 'cv.json'), self.seed, self._cv_status)
        self.writer.close()
        if self.verbose:
            print('Fold [{}/{}] completed'.format(self.i_fold + 1, self.config['experiment']['n_folds']))
        self.i_fold += 1
        self.i_epoch = 0

    def open_epoch(self):
        pass
        # self.track_metric = ((self.i_epoch + 1) % self.period_metric == 0) if self.period_metric else False

    def close_epoch(self):
        self.i_epoch += 1
