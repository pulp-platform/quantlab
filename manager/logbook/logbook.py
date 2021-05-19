import sys
import os
import json
import importlib
import random

from .logsmanager import LogsManager
from manager.assistants import QuantLabLibrary

import manager.assistants
from typing import Union


_MAX_SEED = 2**32


class Logbook(object):

    def __init__(self, problem: str, topology: str, verbose: bool = False):
        """The entity that coordinates experimental designs, units, and runs.

        Args:
            problem: the name of the data set that you are trying to solve.
            topology: the name of the DNN topology (of family of topologies)
                that will be used to solve the problem.
            verbose: whether to print console QuantLab messages.

        Attributes:
            _verbose (str): whether to print console QuantLab messages.
            _path_qlhome (str): the path to QuantLab's home directory.
            _plib (:py:class:`manager.assistants.QuantLabLibrary`): the
                problem-specific but topology-agnostic collection of Python
                software abstractions required to assemble the machine
                learning system.
            _tlib (:py:class:`manager.assistants.QuantLabLibrary`): the
                topology-specific collection of Python software abstractions
                required to assemble the machine learning system.
            config (dict): the functional description of an experimental
                unit; the logbook will exchange this information with the
                assistants to assemble the required machine learning system.
            logs_manager (LogsManager): the entity that manages the logging
                interactions with the disk; in multi-process runs, only the
                master process will have one.
        """

        self._verbose = verbose
        self._path_qlhome = sys.path[0]

        self._plib = None
        self._tlib = None
        self._load_libraries(problem, topology)

        self.config       = None
        self.logs_manager = None  # [MASTER-ONLY]

    # === ML SYSTEM LIBRARIES ===
    def _load_libraries(self, problem: str, topology: str) -> None:
        self._plib = QuantLabLibrary(importlib.import_module('.'.join(['systems', problem])))
        self._tlib = QuantLabLibrary(importlib.import_module('.'.join(['systems', problem, topology])))
        assert self._plib.name == problem
        assert self._tlib.name == topology

    @property
    def problem(self):
        return self._plib.name

    @property
    def topology(self):
        return self._tlib.name

    # === LOGS MANAGER [MASTER-ONLY] ===
    @property
    def path_logs(self) -> str:
        return os.path.join(self._path_qlhome, 'systems', self.problem, self.topology, 'logs')

    def boot_logs_manager(self, exp_id: Union[int, None] = None) -> None:  # in multi-process settings, only the master process should call this method
        """Create the entity to handle logging transactions with the disk.

        To avoid race conditions in multi-process experimental runs, QuantLab
        uses a master-worker pattern to handle disk transactions. Hence, in
        multi-process settings, only the master process will own a copy of
        this entity (and should call this function).
        """

        if exp_id is None:
            experimental_units = [f for f in os.listdir(self.path_logs) if f.startswith('exp')]
            exp_id = 0 if len(experimental_units) == 0 else (max(int(f.replace('exp', '')) for f in experimental_units) + 1)

        self.logs_manager = LogsManager(self.path_logs, exp_id)

    def create_config(self,
                      target_loss: str, ckpt_period: int,
                      n_folds: int, cv_seed: int,
                      # fix_sampler: bool, sampler_seed: int,
                      fix_network: bool, network_seed: int) -> None:
        """Create a private experimental unit configuration from a public one.

        QuantLab allows to specify the build of a DNN learning system in a
        functional fashion via a configuration dictionary. This description is
        meant to provide the missing pieces needed to instantiate a full DNN
        learning system using the building blocks implemented in the problem
        and topology sub-modules.

        To facilitate the life of the user, each topology sub-package stores a
        *public* ``config.json`` file that can be edited manually before
        running the configuration flow. When this function is invoked, the
        ``Logbook`` will read the JSON file into a Python dictionary, add
        some information (both user-specified and stochastically-generated),
        and finally instruct the ``LogsManager`` to store the private copy to
        the experimental unit's logs folder. An example of user-specified
        property is the checkpointing frequency (you might be wasting your
        disk space by storing hundreds of checkpoints for runs that take just
        a few hours on a laptop, whereas crashes on large distributed clusters
        might cost you days and even money). Examples of
        stochastically-generated properties are seeds, which will ensure
        consistent data set rebuilds in case of crashes (preventing validation
        points from filtering into the training set), (possibly) replicability
        of your results and (hopefully) statistically sound results (you can
        anyway cheat and go editing the private copy of the JSON: enjoy your
        cherry-picking!).

        Args:
            target_loss: whether to optimise training or validation loss.
            ckpt_period: the frequency (in epochs) at which checkpoints will
                be saved.
            n_folds: the number of cross-validation folds.
            cv_seed: the seed for the algorithm that splits training data into
                folds.
            fix_network: whether to initialise the network in the same way
                across folds or not.
            network_seed: the seed for the network initialisation algorithm in
                case the network must be initialised in the same way across
                different folds.
        """

        assert cv_seed      < _MAX_SEED
        # assert sampler_seed < _MAX_SEED
        assert network_seed < _MAX_SEED

        # load shared configuration
        config_file = os.path.join(self._path_qlhome, 'systems', self.problem, self.topology, 'config.json')
        with open(config_file, 'r') as fp:
            self.config = json.load(fp)

        # add experiment-specific details
        self.config['experiment']                       = {}
        self.config['experiment']['target_loss']        = target_loss
        self.config['experiment']['ckpt_period']        = ckpt_period

        self.config['data']['dataset']['cv']            = {}
        self.config['data']['dataset']['cv']['n_folds'] = n_folds
        self.config['data']['dataset']['cv']['seed']    = cv_seed if cv_seed >= 0 else random.randint(0, _MAX_SEED)

        # self.config['data']['sampler']                  = {}
        # # if fix_sampler:
        # #     if sampler_seed >= 0:
        # #         sampler_seeds = [sampler_seed] * n_folds
        # #     else:
        # #         sampler_seeds = [random.randint(0, _MAX_SEED)] * n_folds
        # # else:
        # #     sampler_seeds = [random.randint(0, _MAX_SEED) for _ in range(0, n_folds)]
        # self.config['data']['sampler']['seeds']         = [sampler_seed if sampler_seed >= 0 else random.randint(0, _MAX_SEED)] * n_folds if fix_sampler else [random.randint(0, _MAX_SEED) for _ in range(0, n_folds)]

        self.config['network']['seeds']                 = [network_seed if network_seed >= 0 else random.randint(0, _MAX_SEED)] * n_folds if fix_network else [random.randint(0, _MAX_SEED) for _ in range(0, n_folds)]

        # TODO: validate configuration using JSON Schema (Python package ``jsonschema``)

    # === ASSISTANTS MESSAGES PREPARATION ===
    @property
    def path_data(self) -> str:
        return os.path.join(self._path_qlhome, 'systems', self.problem, 'data')

    def send_datamessage(self) -> manager.assistants.DataMessage:
        return manager.assistants.DataMessage(self.path_data, self.config['data'], self._tlib)

    def send_networkmessage(self) -> manager.assistants.NetworkMessage:
        return manager.assistants.NetworkMessage(self.config['network'], self._tlib)

    def send_trainingmessage(self) -> manager.assistants.TrainingMessage:
        return manager.assistants.TrainingMessage(self.config['training'], self._tlib)

    def send_metermessage(self) -> manager.assistants.MeterMessage:
        return manager.assistants.MeterMessage(self.n_epochs, self.config['meters'], self._tlib, self._plib)

    # === PROPERTIES OF THE EXPERIMENTAL RUN'S TRAINING FLOW ===
    @property
    def n_folds(self) -> int:
        return self.config['data']['dataset']['cv']['n_folds']

    @property
    def n_epochs(self) -> int:
        return self.config['training']['n_epochs']

    # === CHECKPOINTING CRITERIA ===
    @property
    def target_loss(self) -> str:
        return self.config['experiment']['target_loss']

    @property
    def ckpt_period(self) -> int:
        return self.config['experiment']['ckpt_period']
