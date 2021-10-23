# 
# cvsplit.py
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
import itertools

from typing import Tuple, List


def default_dataset_cv_split(dataset: torch.utils.data.Dataset,
                             n_folds: int,
                             current_fold_id: int,
                             cv_seed: int) -> Tuple[List[int], List[int]]:
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
        n_folds: the size of the partition.
        current_fold_id: the index of the partition element that plays the
            role of the validation set for the current fold.
        cv_seed: the seed for the random number generator (RNG); it ensures
            that the partition is consistent amongst multiple sessions of the
            experimental run, and that a fold's validation points do not leak
            into the fold's training set (e.g., in case of crashed/interrupted
            runs that require restoring the last checkpointed state).

    Returns:
        (tuple): tuple containing:

            train_fold_indices: the indices of the files representing the
                fold's training set.
            valid_fold_indices: the indices of the files representing the
                fold's validation set.

    """

    torch.manual_seed(cv_seed)

    # partition the set of indices
    indices = torch.randperm(len(dataset)).tolist()
    folds_indices = []
    for fold_id in range(n_folds):
        folds_indices.append(indices[fold_id::n_folds])

    # get the fold's indices partitions
    train_fold_indices = list(itertools.chain(*[folds_indices[i] for i in range(len(folds_indices)) if i != current_fold_id]))
    valid_fold_indices = folds_indices[current_fold_id]

    return train_fold_indices, valid_fold_indices

