# 
# data.py
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

import torch
import torch.utils.data

from .library import QuantLabLibrary
from manager.platform import PlatformManager


__all__ = [
    'DataMessage',
    'DataAssistant',
]


class DataMessage(object):

    def __init__(self, path_data: str, cv_config: dict, data_config: dict, library: QuantLabLibrary) -> None:
        """Describe how to build :obj:`torch.utils.data.DataLoader`s.

        An object of this class implements the server-side of a *dependency
        injection* pattern whose client-side is a :obj:`DataAssistant` object.

        Args:
            path_data: the path to the problem's data set (both training and
                validation/test points).
            config: the functional description of pre-processing,
                cross-validation, and batching.
            library: the collection of class and function definitions that
                allow to assemble a ``DataLoader``.

        """

        self._path_data   = path_data
        self._cv_config   = cv_config
        self._data_config = data_config
        self._library     = library

    @property
    def path_data(self):
        return self._path_data

    @property
    def cv_config(self):
        return self._cv_config

    @property
    def data_config(self):
        return self._data_config

    @property
    def library(self):
        return self._library


class DataAssistant(object):

    def __init__(self, partition: str):
        """The entity that assembles :obj:`torch.utils.data.DataLoader`s.

        An object of this class implements the client-side of a *dependency
        injection* pattern whose server-side is a :obj:`DataMessage` object.
        QuantLab assumes that ``DataMessage``s are created by a :obj:`Logbook`
        instance according to the machine learning system's library and on the
        experimental unit's configuration.

        This class follows the *builder* pattern. A simple *factory* pattern
        is not sufficient because a ``DataLoader`` is too complex an object to
        be built during a single call to a simple construction method. See
        below for a more detailed descriptions of the steps required by the
        build.

        To understand the working of this entity, you should be familiar with
        PyTorch's data management abstraction: the ``DataLoader``.
        ``DataLoader`` objects are Python :obj:`iterator`s composed of three
        main sub-systems:
            * a :obj:`torch.utils.data.Dataset` object;
            * a :obj:`torch.utils.data.Sampler` object;
            * a :obj:`multiprocessing.Queue` of worker processes.
        In turn, a ``Dataset`` consists of:
            * a mapping from integers to filepaths;
            * a transform, i.e., a pipeline consisting of one or more
              pre-processing functions that should be applied to the raw file
              before feeding it to a :obj:`torch.nn.Module` object.
        It is also beneficial to point out that when the ``Sampler`` is
        created it is passed the list of integer indices of the ``Dataset``'s
        mapping. Whenever the ``DataLoader`` is queried for a batch of data
        points, it performs the following operations:
            * it pops a free worker from the queue;
            * the worker queries the ``Sampler`` for a list of indices (the
              size of this list is the batch size);
            * the worker uses the ``Dataset``'s mapping from integers to
              filepaths to retrieve the files;
            * the worker applies the pipeline of pre-processing functions
              specified by the ``Dataset``'s transform to each retrieved file;
              the worker can also apply pre-processing to labels;
            * the worker assembles the pre-processed files into a
              multi-dimensional array where the first dimension indexes the
              batch dimension, i.e., a :obj:`torch.Tensor`; if this assembly
              is non-trivial, the ``Dataset`` can specify an optional method
              ``collate_fn``;
            * the worker notifies the ``DataLoader`` that the batch is ready,
              and is pushed back into the queue of free workers;
            * the ``DataLoader`` returns the batch to the calller program.

        This entity decomposes the assembly of a ``DataLoader`` in three parts:
            * ``DataSet`` creation;
            * ``Sampler`` creation;
            * ``DataLoader`` creation.
        QuantLab trades part of PyTorch's configurability against giving the
        user the opportunity to focus more on the functional aspects of the
        deep learning system's configuration. In fact, QuantLab limits its
        choice of ``Sampler``s to :obj:`torch.utils.data.RandomSampler` and
        :obj:`torch.utils.data.SequentialSampler` for single-process runs, and
        to :obj:`torch.utils.data.distributed.DistributedSampler` for
        multi-process (Horovod) runs. Moreover, it tries to automatically
        determine the optimal number of worker processes and whether to pin
        memory to processes (which might be useful in NUMA computing nodes).
        Hence, all that is left to the user to specify are the functional
        and DNN-topology-related aspects of the data configuration:
            * the transforms, i.e., the pre-processing functions that can be
              applied individually to each data point;
            * the ``Dataset``'s constructor functions, since this might depend
              on the specific format of the data set files; it is at this
              stage that cross-validation should be taken into account;
            * the batch sizes for both training and validation/test sets.

        On some data sets, performing multi-fold cross-validation experimental
        runs might require additional care when splitting the training set in
        a training fold and a validation one. For example, data sets recording
        healthcare data about several patients might include multiple data
        points describing the same individual, and a split which sends parts
        of these points in the training fold and part of them in the
        validation fold might bias the results. Hence, ``DataAssistant``s
        provide an overridable default cross-validation splitting function
        that the user can replace with a custom splitting function.

        Arguments:
            partition: whether this assistant will be in charge of creating
                the training, validation or test ``DataLoader``.

        Attributes:
            _transform_fun(Callable[..., torch.Tensor]): the function
                implementing the pre-processing of a data point, possibly
                including its label.
            _transform_kwargs (dict): the keyword arguments that specify how
                to instantiate the data-preprocessing function.
            _load_data_set_fun (Callable[[str, Callable[..., torch.Tensor], bool], torch.utils.data.Dataset]):
                the function to create ``Dataset``s; it should be passed the
                path to the data files, the transform function, and whether to
                create the training or the validation/test ``Dataset``.
            _path_data (str): the path to the problem's database (training,
                validation and test points).
            _n_folds (int): the number of cross-validation folds of the
                current experimental unit.
            _fold_id (int): the identifier of the current cross-validation
                fold.
            _cv_seed (int): the seed for PyTorch's random number generator;
                this is meant to ensure consistency of the splits in case they
                need to be recomputed (e.g., after an experiment crashes or is
                interrupted).
            _bs (int): the size of the ``DataLoader``'s batch.
        """

        self._partition = partition

        # database
        self._path_data            = None
        # ingredients for dataset creation
        self._load_data_set_fun    = None
        self._load_data_set_kwargs = None
        # cross-validation
        self._n_folds              = None
        self._fold_id              = None
        self._cv_seed              = None
        # transform
        self._transform_class      = None
        self._transform_kwargs     = None

        # ingredients for sampler creation
        self._sampler_seeds = None

        # ingredients for dataloader creation
        self._bs = None

    def recv_datamessage(self, datamessage: DataMessage) -> None:
        """Resolve the functional dependencies for the assembly.

        Args:
            datamessage: the collection of dependencies that the
                ``DataAssistant`` should be aware of when assembling the
                ``DataLoader``s.

        """

        # ``Dataset`` - data location on the filesystem
        self._path_data = datamessage.path_data

        # ``Dataset`` - import function (mandatory, but ``kwargs`` are optional)
        self._load_data_set_fun = getattr(datamessage.library.module, 'load_data_set')  # the `load_data_set` function MUST be implemented by EACH topology sub-package
        try:
            self._load_data_set_kwargs = datamessage.data_config['dataset']['load_data_set']['kwargs']
        except KeyError:
            self._load_data_set_kwargs = {}

        # ``Dataset`` - cross-validation details (mandatory, but ``dataset_cv_split_fun`` is optional)
        self._n_folds = datamessage.cv_config['n_folds']
        self._cv_seed = datamessage.cv_config['seed']

        # ``Dataset`` - pre-processing functions (mandatory)
        self._transform_class  = getattr(datamessage.library.module, datamessage.data_config['dataset']['transform']['class'])
        self._transform_kwargs = datamessage.data_config['dataset']['transform']['kwargs']

        # ``Sampler`` - seed
        try:
            self._sampler_seeds = datamessage.data_config['sampler']['seeds']
        except KeyError:
            pass

        # ``DataLoader`` - batch sizes (mandatory)
        self._bs = datamessage.data_config['dataloader']['bs']

    def get_dataset(self) -> torch.utils.data.Dataset:

        transform = self._transform_class(**self._transform_kwargs)
        dataset   = self._load_data_set_fun(self._partition, self._path_data, n_folds=self._n_folds, current_fold_id=self._fold_id, cv_seed=self._cv_seed, transform=transform, **self._load_data_set_kwargs)

        return dataset

    def get_sampler(self,
                    platform: PlatformManager,
                    dataset: torch.utils.data.Dataset) -> torch.utils.data.Sampler:

        if platform.is_horovod_run:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=platform.global_size, rank=platform.global_rank, shuffle=True if self._partition == 'train' else False, seed=self._sampler_seeds[self._fold_id] if self._partition == 'train' else 0)
        else:
            generator = torch.Generator()
            generator.manual_seed(self._sampler_seeds[self._fold_id] if self._partition == 'train' else 0)
            sampler = torch.utils.data.RandomSampler(dataset, generator=generator) if self._partition == 'train' else torch.utils.data.SequentialSampler(dataset)

        return sampler

    def get_dataloader(self,
                       platform: PlatformManager,
                       dataset: torch.utils.data.Dataset,
                       sampler: torch.utils.data.Sampler) -> torch.utils.data.DataLoader:

        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=self._bs,
                                             sampler=sampler,
                                             collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None,
                                             num_workers=platform.num_workers,
                                             pin_memory=platform.pin_memory)

        return loader

    def prepare(self, platform: PlatformManager, fold_id: int) -> torch.utils.data.DataLoader:
        """Create the training and validation/test ``DataLoader``s.

        Args:
            platform: the entity that registers the engineering aspects of the
                computation: hardware specifications, OS details, MPI
                configuration (via Horovod).
            fold_id: the identifier of the fold for which to prepare the
                ``DataLoader``; in cross-validation experiment, this
                determines the partition that will play the role of validation
                set.

        Returns:
            loader: the ``DataLoader``.
        """

        self._fold_id = fold_id

        dataset = self.get_dataset()
        sampler = self.get_sampler(platform, dataset)
        loader  = self.get_dataloader(platform, dataset, sampler)

        return loader
