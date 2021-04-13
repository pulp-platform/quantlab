import os
import importlib
import torchvision


def load_data_sets(logbook):

    pipesmod  = importlib.import_module('.'.join(['', 'preprocess', logbook.config['data']['preprocess']]), package=logbook.lib.__name__)
    pipelines = getattr(pipesmod, 'get_pipelines')(logbook.config['data']['augment'])

    train_set = torchvision.datasets.ImageFolder(os.path.join(logbook.dir_data, 'train'), pipelines['training'])
    valid_set = torchvision.datasets.ImageFolder(os.path.join(os.path.realpath(logbook.dir_data), 'val'), pipelines['validation'])

    return train_set, valid_set
