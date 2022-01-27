import torch
import torchvision

from .helper import parameter_generation
from .helper import AudioProcessor
from .helper import AudioGenerator

from .helper import GSCToTensor


__all__ = [
    'load_data_set',
    'GSCToTensor',
]


def load_data_set(partition: str,
                  path_data: str,
                  n_folds: int,
                  current_fold_id: int,
                  cv_seed: int,
                  transform: torchvision.transforms.Compose) -> torch.utils.data.Dataset:

    training_parameters, data_preprocessing_parameters = parameter_generation()
    ap = AudioProcessor(path_data, training_parameters, data_preprocessing_parameters)

    if partition == 'train':
        mode = 'training'
    elif partition == 'valid':
        mode = 'validation'
    elif partition == 'test':
        mode = 'testing'
    else:
        raise ValueError

    dataset = AudioGenerator(mode, ap, training_parameters)

    return dataset
