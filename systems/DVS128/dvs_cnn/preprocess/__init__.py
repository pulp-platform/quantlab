import torch
import torchvision

from .augment_transform import DVSAugmentTransform
from systems.DVS128 import DVS128DataSet
__all__ = [
    'load_data_set',
    'DVSAugmentTransform'
]

def load_data_set(partition: str,
                  path_data: str,
                  n_folds: int,
                  current_fold_id: int,
                  cv_seed: int,
                  transform: torchvision.transforms.Compose,
                  n_val_subjects : int,
                  **kwargs) -> torch.utils.data.Dataset:

    # **kwargs is passed directly to DVS128DataSet and should contain:
    # - cnn_win
    # - tcn_win
    # - window_stride

    # for k-fold cross validation, take n_folds different disjoint sets of
    # n_val_subjects subjects as the validation set, starting with the last
    # n_val_subjects subjects. One fold (the last one) may have less than
    # n_val_subjects subjects in the validation set.
    assert 30-(n_folds-1)*n_val_subjects > 0, f"Can't have {n_folds} folds with {n_val_subjects} subjects in the validation set!"

    # if number of subjects is not divisible by n_val_subjects, the last fold will
    # contain fewer subjects. the number of subjects is a prime number so this
    # will always be the case if we do "exhaustive" CV.
    start_user = max(0, 30-(current_fold_id+1)*n_val_subjects)
    val_subjects = list(range(start_user, 30-current_fold_id*n_val_subjects))
    train_subjects = [i for i in range(1, 30) if i not in val_subjects]
    file_suffix = "_lo"+"".join([f"_{u}" for u in val_subjects]) + f"_{partition}"
    if partition == "train":
        include_subjects = train_subjects
    else:
        assert partition == "valid", f"load_data_set(): Invalid partition {partition}!"
        include_subjects = val_subjects


    dataset = DVS128DataSet(data_dir=path_data, include_subjects=include_subjects, single_out=True, file_suffix=file_suffix, transform=transform, **kwargs)
    return dataset
