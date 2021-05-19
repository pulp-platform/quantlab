# -*- coding: utf-8 -*-
import torch
import torch.utils.data
import itertools

from typing import Tuple, List, Callable

from .library import QuantLabLibrary
from manager.platform import PlatformManager


__all__ = [
    'DataMessage',
    'DataAssistant',
]


class DataMessage(object):

    def __init__(self, path_data: str, config: dict, library: QuantLabLibrary) -> None:
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

        self._path_data = path_data
        self._config    = config
        self._library   = library

    @property
    def path_data(self):
        return self._path_data

    @property
    def config(self):
        return self._config

    @property
    def library(self):
        return self._library


class DataAssistant(object):

    def __init__(self):
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

        Attributes:
            _transform_fun(Callable[..., torch.Tensor]): the function
                implementing the pre-processing of a data point, possibly
                including its label.
            _transform_kwargs (dict): the keyword arguments that specify how
                to instantiate the data-preprocessing function.
            _path_data (str): the path to the problem's data set (both
                training and validation/test points).
            _dataset_load_fun (Callable[[str, Callable[..., torch.Tensor], bool], torch.utils.data.Dataset]):
                the function to create ``Dataset``s; it should be passed the
                path to the data files, the transform function, and whether to
                create the training or the validation/test ``Dataset``.
            _seed (int): the seed for PyTorch's random number generator; this
                is meant to ensure consistency of the splits in case they need
                to be recomputed (e.g., after an experiment crashes or is
                interrupted).
            _n_folds (int): the number of cross-validation folds of the
                current experimental unit.
            _fold_id (int): the identifier of the current cross-validation
                fold.
            _dataset_cv_split_fun(Callable[[torch.utils.data.Dataset, int, int, int], Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]):
                the function performing the split of the indices of the data
                points.
            _bs_train (int): the size of the training ``DataLoader``'s batch.
            _bs_valid (int): the size of the validation/test ``DataLoader``'s
                batch.

        """

        # ingredients for dataset creation
        # transform
        self._transform_fun    = None
        self._transform_kwargs = None
        # dataset
        self._path_data        = None
        self._dataset_load_fun = None
        # cross-validation
        self._n_folds              = None
        self._fold_id              = None
        self._cv_seed              = None
        self._dataset_cv_split_fun = None

        # ingredients for dataloader creation
        self._bs_train = None
        self._bs_valid = None

    def recv_datamessage(self, datamessage: DataMessage) -> None:
        """Resolve the functional dependencies for the assembly.

        Args:
            datamessage: the collection of dependencies that the
                ``DataAssistant`` should be aware of when assembling the
                ``DataLoader``s.

        """

        # ``Dataset`` - pre-processing functions (mandatory)
        self._transform_fun    = getattr(datamessage.library.module, datamessage.config['dataset']['transforms']['function'])
        self._transform_kwargs = datamessage.config['dataset']['transforms']['kwargs']

        # ``Dataset`` - import function (mandatory)
        self._path_data        = datamessage.path_data
        self._dataset_load_fun = getattr(datamessage.library.module, 'dataset_load')  # the `dataset_load` function MUST be implemented by EACH topology sub-package

        # ``Dataset`` - cross-validation details (mandatory, but ``dataset_cv_split_fun`` is optional)
        self._n_folds              = datamessage.config['dataset']['cv']['n_folds']
        self._cv_seed              = datamessage.config['dataset']['cv']['seed']
        self._dataset_cv_split_fun = getattr(datamessage.library.module, 'dataset_cv_split') if hasattr(datamessage.library.module, 'dataset_cv_split') else self.default_dataset_cv_split

        # # ``Sampler`` - seed
        # self._sampler_seeds = datamessage.config['sampler']['seeds']

        # ``DataLoader`` - batch sizes (mandatory)
        self._bs_train = datamessage.config['dataloader']['bs']['train']
        self._bs_valid = datamessage.config['dataloader']['bs']['valid']

    @property
    def is_cv_run(self):
        return (self._n_folds > 1) and (self._fold_id is not None)

    def set_fold_id(self, fold_id):
        self._fold_id = fold_id

    @staticmethod
    def default_dataset_cv_split(dataset: torch.utils.data.Dataset,
                                 n_folds: int,
                                 current_fold_id: int,
                                 seed: int) -> Tuple[List[int], List[int]]:
        """Partition a data set into two cross-validation fold data sets.

        Given a data set :math:`\mathcal{D}` containing :math:`N` points
        (so that we can identify it with the set :math:`\{ 0, \dots, N-1 \}`),
        a :math:`K`-fold cross-validation setup (where :math:`K > 1`), and a
        fold index :math:`\bar{k} \in \{ 0, \dots, K-1 \}`, this function
        computes a :math:`K`-partition
        :math:`\{ \mathcal{D}^{(k)} \subset \mathcal{D} \}_{k = 1, \dots, K}`
        of :math:`\{ 0, \dots, N-1 \}` and returns a pair
        :math:`(\mathcal{D}^{(\bar{k})}_{train}, \mathcal{D}^{(\bar{k})}_{valid})`
        of subsets of :math:`\mathcal{D}` such that
        :math:`\mathcal{D}^{(\bar{k})}_{train} \cup \mathcal{D}^{(\bar{k})}_{valid} = \mathcal{D}`
        and
        :math:`\mathcal{D}^{(\bar{k})}_{train} \cap \mathcal{D}^{(\bar{k})}_{valid} = \emptyset`,
        where
        :math:`\mathcal{D}^{(\bar{k})}_{train} = \cup_{k \neq \bar{k}} \mathcal{D}^{(k)}`
        and :math:`\mathcal{D}^{(\bar{k})}_{valid} = \mathcal{D}^{(\bar{k})}`.

        Args:
            dataset: the data set to be split into folds.
            seed (int): the seed for the random number generator (RNG); it
                ensures that the partition is consistent amongst multiple
                parts of the experimental run, and that a fold's validation
                points do not leak into the fold's training set (e.g., in case
                of multi-process runs where each process individually computes
                the partition, or in crashed/interrupted runs which require
                restoring the last checkpointed state).
            n_folds: the size of the partition.
            current_fold_id: the index of the partition element that plays the
                role of the validation set for the current fold.

        Returns:
            (tuple): tuple containing:

                train_fold_indices: the indices of the files representing the
                    fold's training set.
                valid_fold_indices: the indices of the files representing the
                    fold's validation set.

        """

        torch.manual_seed(seed)

        # partition the set of indices
        indices = torch.randperm(len(dataset)).tolist()  # TODO: I also `shuffle` the `DistributedSampler` later; am I sure that this step is necessary (if not an hazard)?
        folds_indices = []
        for fold_id in range(n_folds):
            folds_indices.append(indices[fold_id::n_folds])

        # get the fold's indices partitions
        train_fold_indices = list(itertools.chain(*[folds_indices[i] for i in range(len(folds_indices)) if i != current_fold_id]))
        valid_fold_indices = folds_indices[current_fold_id]

        return train_fold_indices, valid_fold_indices

    def get_dataset(self, train: bool = True) -> torch.utils.data.Dataset:

        transform = self._transform_fun(train=train, **self._transform_kwargs)
        dataset   = self._dataset_load_fun(self._path_data, transform, train=train)

        return dataset

    def get_sampler(self,
                    platform: PlatformManager,
                    dataset: torch.utils.data.Dataset,
                    train: bool = True) -> torch.utils.data.Sampler:

        if platform.is_horovod_run:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=platform.global_size, rank=platform.global_rank, shuffle=train)  # TODO: PyTorch 1.5.0 does not allow to seed ``DistributedSampler``s
        else:
            # generator = torch.Generator()
            # generator.manual_seed(self._sampler_seeds[self._fold_id])  # TODO: PyTorch 1.5.0 does not allow to seed ``RandomSampler``s
            sampler = torch.utils.data.RandomSampler(dataset) if train else torch.utils.data.SequentialSampler(dataset)

        return sampler

    def get_dataloader(self,
                       platform: PlatformManager,
                       dataset: torch.utils.data.Dataset,
                       sampler: torch.utils.data.Sampler,
                       train: bool = True) -> torch.utils.data.DataLoader:

        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=self._bs_train if train else self._bs_valid,
                                             sampler=sampler,
                                             collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None,
                                             num_workers=platform.num_workers,
                                             pin_memory=platform.pin_memory)

        return loader

    def prepare(self, platform: PlatformManager, fold_id: int) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Create the training and validation/test ``DataLoader``s.

        Args:
            platform: the entity that registers the engineering aspects of the
                computation: hardware specifications, OS details, MPI
                configuration (via Horovod).
            fold_id: the identifier of the fold for which to prepare the
                ``DataLoader``s; in cross-validation experiment, this
                determines the partition that will play the validation set
                role.

        Returns:
            (tuple):

                train_loader: the ``DataLoader`` for training data.
                valid_loader: the ``DataLoader`` for validation (test) data.

        """

        self.set_fold_id(fold_id)

        # build the datasets (taking into account cross-validation, if required by the experimental unit)
        if self.is_cv_run:

            dataset = self.get_dataset(train=True)

            train_fold_indices = None
            valid_fold_indices = None

            # master-workers synchronisation point: the master computes the indices of the files in the folds, and distributes these indices to worker processes
            if (not platform.is_horovod_run) or platform.is_master:
                train_fold_indices, valid_fold_indices = self._dataset_cv_split_fun(dataset, self._n_folds, self._fold_id, self._cv_seed)
            if platform.is_horovod_run:
                train_fold_indices = platform.hvd.broadcast_object(train_fold_indices, root_rank=platform.master_rank, name='train_fold_indices')
                valid_fold_indices = platform.hvd.broadcast_object(valid_fold_indices, root_rank=platform.master_rank, name='valid_fold_indices')

            # build the fold's datasets
            train_dataset = torch.utils.data.Subset(dataset, train_fold_indices)
            valid_dataset = torch.utils.data.Subset(dataset, valid_fold_indices)

        else:

            train_dataset = self.get_dataset(train=True)
            valid_dataset = self.get_dataset(train=False)

        # build the samplers
        train_sampler = self.get_sampler(platform, train_dataset, train=True)
        valid_sampler = self.get_sampler(platform, valid_dataset, train=False)

        # build the dataloaders
        train_loader = self.get_dataloader(platform, train_dataset, train_sampler, train=True)
        valid_loader = self.get_dataloader(platform, valid_dataset, valid_sampler, train=False)

        return train_loader, valid_loader
