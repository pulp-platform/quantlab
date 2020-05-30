import sys
import os
import shutil

import horovod.torch as hvd
import multiprocessing as mp
from collections import OrderedDict

import importlib
import json
import math
import torch
from torch.utils.tensorboard import SummaryWriter


__MASTER_PROC_RANK__ = 0

__MAX_EXPERIMENTS__ = 1000
__ALIGN_EXP__ = math.ceil(math.log10(__MAX_EXPERIMENTS__))  # experiment ID string length (decimal literal)
__MAX_SEED__ = 10000
__MAX_EPOCHS__ = 10000
__ALIGN_EPOCHS__ = math.ceil(math.log10(__MAX_EPOCHS__))  # checkpoint ID string length (decimal literal)
__MAX_CV_FOLDS__ = 10
__ALIGN_CV_FOLDS__ = math.ceil(math.log10(__MAX_CV_FOLDS__))  # cross-validation fold ID string length (decimal literal)


class Logbook(object):

    def __init__(self, problem, topology, exp_id, verbose=True):
        """Experiment management abstraction.

        The logbook registers the information needed by the *assistants* to
        instantiate data sets, network and training procedure, plus the status
        of the current experiment.

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
            dir_exp (str): The full path to the log folder.
                This path actually passes through a symbolic link to the real
                log folder, which should have been created on a high-capacity
                storage device (e.g., a HDD) to store a large number of
                PyTorch checkpoint files and TensorBoard event files.
            config (:obj:`dict`): The full description of the experiment.
            dir_saves (str): The full path to the checkpoints folder.
                PyTorch checkpoints about the current fold will be stored
                here.
            dir_stats (str): The full path to the statistics folder.
                TensorBoard events about the current fold will be stored here.
            writer (:obj:`SummaryWriter`): The object that logs the experiment
                configuration and statistics to a TensorBoard event file.
            ckpt (None or :obj:`dict`): The selected checkpoint from which the
                experiment should be resumed.
            best_metric (float): The best result achieved by the experiment up
                to the current iteration.
            i_epoch (int): The current iteration of the experiment.
            meter (:obj:`Meter`): The object which measures loss and data set-
                specific metrics.
            verbose (bool): Whether status messages should be printed.

        """
        # get process information
        self.hw_cfg        = None
        self.sw_cfg        = None
        self.is_master     = None
        self.verbose       = None
        self._whoami(verbose)
        # get experiment general information
        self.problem       = problem
        self.topology      = topology
        self.lib           = importlib.import_module('.'.join(['problems', self.problem, self.topology]))
        self.dir_data      = None
        self.dir_exp       = None
        self.config        = None
        self._setup_experiment(exp_id)

        # experiment status
        self.i_fold        = None
        # logging instrumentation
        self.dir_saves     = None
        self.dir_stats     = None
        self.writer        = None
        self.i_epoch       = None
        self.meter         = None
        self.metrics       = None
        self.target_metric = None

    def _whoami(self, verbose):
        hw_cfg = {
            'n_cpus': mp.cpu_count(),
            'n_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        sw_cfg = {
            'global_size': hvd.size(),
            'global_rank': hvd.rank(),
            'local_size':  hvd.local_size(),
            'local_rank':  hvd.local_rank(),
            'master_rank': __MASTER_PROC_RANK__
        }
        assert sw_cfg['local_size'] <= hw_cfg['n_cpus']  # maximum one process per core
        assert hw_cfg['n_gpus'] <= hw_cfg['n_cpus']        # maximum one GPU per core
        if hw_cfg['n_gpus'] > 0:
            assert sw_cfg['local_size'] <= hw_cfg['n_gpus']  # maximum one process per GPU
            torch.cuda.set_device(sw_cfg['local_rank'])      # if node is equipped with GPUs, each process should be pinned to one
            device = torch.cuda.current_device()
        else:
            device = torch.device('cpu')
        hw_cfg['device'] = device

        self.hw_cfg = hw_cfg
        self.sw_cfg = sw_cfg
        self.is_master = self.sw_cfg['global_rank'] == self.sw_cfg['master_rank']
        self.verbose = verbose and self.is_master

    def _setup_experiment(self, exp_id):
        """Get pointers to the data and experiment folders.

        Args:
            exp_id (str): The decimal literal identifying the experiment.

        """
        if self.is_master:
            QUANT_HOME = sys.path[0]
            # get pointers to HARD SHARED resources
            HARD_STORAGE = os.path.join(QUANT_HOME, 'cfg', 'hard_storage.json')
            with open(HARD_STORAGE, 'r') as fp:
                d = json.load(fp)
                # data
                HARD_HOME_DATA = os.path.join(d['data'], 'Quant')
                HARD_DIR_DATA = os.path.join(HARD_HOME_DATA, 'problems', self.problem, 'data')
                if not os.path.isdir(HARD_DIR_DATA):
                    raise FileNotFoundError('{} hard directory (data) not found: {}'.format(self.problem, HARD_DIR_DATA))
                # log
                HARD_HOME_LOGS = os.path.join(d['logs'], 'Quant')
                HARD_DIR_LOGS = os.path.join(HARD_HOME_LOGS, 'problems', self.problem, 'logs')
                if not os.path.isdir(HARD_DIR_LOGS):
                    raise FileNotFoundError('{} hard directory (logs) not found: {}'.format(self.problem, HARD_DIR_LOGS))
            # get pointers to SOFT SHARED resources (which are redirected to HARD ones using symlinks)
            DIR_PROBLEM = os.path.join(QUANT_HOME, 'problems', self.problem)
            dir_data = os.path.join(DIR_PROBLEM, 'data')
            if not os.path.isdir(dir_data):
                os.symlink(HARD_DIR_DATA, dir_data)
            dir_logs = os.path.join(DIR_PROBLEM, 'logs')
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
            dir_exp = os.path.join(dir_logs, 'exp'+str(exp_id).rjust(__ALIGN_EXP__, '0'))
            if not os.path.isdir(dir_exp):
                os.mkdir(dir_exp)

            self.dir_data = dir_data
            self.dir_exp  = dir_exp

            if self.verbose:
                # print setup message
                message  = 'EXPERIMENT LOGBOOK\n'
                message += 'Problem:               {}\n'.format(self.problem)
                message += 'Network topology:      {}\n'.format(self.topology)
                message += 'Data directory:        {}\n'.format(self.dir_data)
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

            # load configuration
            private_config_file = os.path.join(self.dir_exp, 'config.json')

            if not os.path.isfile(private_config_file):
                # no configuration in the experiment folder: look for global one
                shared_config_file = os.path.join(os.path.dirname(self.lib.__file__), 'config.json')
                if not os.path.isfile(shared_config_file):
                    raise FileNotFoundError('Configuration file not found: {}'.format(shared_config_file))
                shutil.copyfile(shared_config_file, private_config_file)
                # generate seed for experiment
                with open(private_config_file, 'r+') as fp:
                    config = json.load(fp)
                    config['experiment']['seed'] = torch.randint(__MAX_SEED__, (1,)).item()
                    fp.seek(0)
                    json.dump(config, fp, indent=4)
                    fp.truncate()

            with open(private_config_file, 'r') as fp:
                self.config = json.load(fp)

        # communicate data pointer and experiment configuration to worker processes
        self.dir_data = hvd.broadcast_object(self.dir_data, root_rank=__MASTER_PROC_RANK__, name='dir_data')
        self.config = hvd.broadcast_object(self.config, root_rank=__MASTER_PROC_RANK__, name='config')

    def _fold_folder_name(self, i_fold):
        return 'fold'+str(i_fold).rjust(__ALIGN_CV_FOLDS__, '0')

    def get_training_status(self):
        if self.is_master:
            # which fold should be resumed (i.e., the last)?
            folds_list = os.listdir(self.dir_exp)
            try:
                i_fold = max([int(f.replace('fold', '')) for f in folds_list])
            except ValueError:
                i_fold = 0
            self.i_fold = i_fold

        # communicate current fold to worker processes
        self.i_fold = hvd.broadcast_object(self.i_fold, root_rank=__MASTER_PROC_RANK__, name='i_fold')

    def open_fold(self):
        if self.is_master:

            # get fold directory
            dir_fold = os.path.join(self.dir_exp, self._fold_folder_name(self.i_fold))
            if not os.path.isdir(dir_fold):
                os.mkdir(dir_fold)

            # get directories for fold checkpoints and logs
            dir_saves = os.path.join(dir_fold, 'saves')
            if not os.path.isdir(dir_saves):
                os.mkdir(dir_saves)
            dir_stats = os.path.join(dir_fold, 'stats')
            if not os.path.isdir(dir_stats):
                os.mkdir(dir_stats)

            # set pointers
            self.dir_saves = dir_saves
            self.dir_stats = dir_stats
            self.writer = SummaryWriter(log_dir=self.dir_stats)

        # create logging instrumentation on every process
        meter_module = importlib.import_module('.'.join(['problems', self.problem, 'meter']))
        self.meter   = getattr(meter_module, 'Meter')(self.lib.postprocess_pr, self.lib.postprocess_gt)
        self.metrics = {
            'train_loss':   torch.tensor(float('Inf')),
            'train_metric': self.meter.start_metric,
            'valid_loss':   torch.tensor(float('Inf')),
            'valid_metric': self.meter.start_metric
        }
        self.target_metric = self.config['experiment']['metrics']['target_metric']
        assert self.target_metric in self.metrics.keys()

    def close_fold(self):
        if self.is_master:
            self.writer.close()
            if self.verbose:
                print('Fold [{}/{}] completed'.format(self.i_fold + 1, self.config['experiment']['n_folds']))
            self.i_fold += 1

        # communicate current fold to worker processes
        self.i_fold = hvd.broadcast_object(self.i_fold, root_rank=__MASTER_PROC_RANK__, name='i_fold_new')

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

    def load_checkpoint(self, net, opt, lr_sched, ctrls, load):
        if self.is_master:

            self.i_epoch = 0

            # look for a checkpoint
            ckpt_list = os.listdir(self.dir_saves)
            if len(ckpt_list) != 0:
                if load == 'best':  # used when evaluating performances
                    ckpt_file = os.path.join(self.dir_saves, 'best.ckpt')
                elif load == 'last':  # used when restoring crashed experiments/resuming interrupted experiments
                    ckpt_file = max([os.path.join(self.dir_saves, f) for f in ckpt_list], key=os.path.getctime)
                else:
                    ckpt_file = None
                # else:  # used in 'WHAT IF' experiments as an initial condition
                #     ckpt_id = str(load).rjust(__ALIGN_EPOCHS__, '0')
                #     ckpt_name = 'epoch' + ckpt_id + '.ckpt'
                #     ckpt_file = os.path.join(self.dir_saves, ckpt_name)

                # load checkpoint from file
                if self.verbose:
                    print('Loading checkpoint {} ...'.format(ckpt_file), end='')
                ckpt = torch.load(ckpt_file)
                if self.verbose:
                    print('done!')

                # restore experiment status
                self.i_epoch = ckpt['fold']['i_epoch']
                self.metrics.update(ckpt['fold']['metrics'])
                net.load_state_dict(ckpt['network'])
                opt.load_state_dict(ckpt['training']['optimizer'])
                lr_sched.load_state_dict(ckpt['training']['lr_scheduler'])
                for c, sd in zip(ctrls, ckpt['training']['quantize']):
                    c.load_state_dict(sd)

        # broadcast experiment status to worker processes
        self.i_epoch = hvd.broadcast_object(self.i_epoch, root_rank=__MASTER_PROC_RANK__, name='i_epoch')

        self.metrics = hvd.broadcast_object(self.metrics, root_rank=__MASTER_PROC_RANK__, name='metrics')

        hvd.broadcast_parameters(net.state_dict(), root_rank=__MASTER_PROC_RANK__)
        hvd.broadcast_optimizer_state(opt, root_rank=__MASTER_PROC_RANK__)
        lr_sched_state_dict = hvd.broadcast_object(lr_sched.state_dict(), root_rank=__MASTER_PROC_RANK__, name='lr_sched_state_dict')
        if not self.is_master:
            lr_sched.load_state_dict(lr_sched_state_dict)
        for i, c in enumerate(ctrls):
            csd = hvd.broadcast_object(c.state_dict(), root_rank=__MASTER_PROC_RANK__, name='controller{}'.format(i))
            if not self.is_master:
                c.load_state_dict(csd)

    def store_checkpoint(self, net, opt, lr_sched, ctrls, is_best=False):
        """Store states of network and training procedure."""
        if self.is_master:

            # build checkpoint
            ckpt = {
                'fold': {
                    'i_epoch': self.i_epoch + 1,
                    'metrics': self.metrics
                },
                'network': net.state_dict(),
                'training': {
                    'optimizer': opt.state_dict(),
                    'lr_scheduler': lr_sched.state_dict(),
                    'quantize': [c.state_dict() for c in ctrls]
                }
            }

            # get checkpoint name
            if is_best:
                ckpt_name = 'best'
            else:
                ckpt_id   = str(self.i_epoch + 1).rjust(__ALIGN_EPOCHS__, '0')
                ckpt_name = 'epoch'+ckpt_id
            ckpt_name = ckpt_name+'.ckpt'

            # store to file
            ckpt_file = os.path.join(self.dir_saves, ckpt_name)
            if self.verbose:
                print('Storing checkpoint {} ...'.format(ckpt_file), end='')
            torch.save(ckpt, ckpt_file)
            if self.verbose:
                print('done!')
