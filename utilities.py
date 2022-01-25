import os
import importlib
import json
import torch.nn as nn

from types import ModuleType
from typing import Tuple, Dict


def import_network(problem: str,
                   topology: str,
                   config_file: str) -> nn.Module:

    def import_libraries(problem: str,
                         topology: str) -> Tuple[ModuleType, ModuleType]:

        topology_pkg = '.'.join(['systems', problem, topology])
        quantize_pkg = '.'.join([topology_pkg, 'quantize'])

        lib = importlib.import_module(topology_pkg)
        qlib = importlib.import_module(quantize_pkg)  # TODO: raise exception if `qlib` does not exist, and return `None` instead

        return lib, qlib

    def import_config(problem: str,
                      topology: str,
                      config_file: str) -> Dict:

        config_path = os.path.join('systems', problem, topology, config_file)

        with open(config_path, 'r') as fp:
            config = json.load(fp)

        return config

    lib, qlib = import_libraries(problem, topology)
    config = import_config(problem, topology, config_file)

    net = getattr(lib, config['network']['class'])(**config['network']['kwargs'])
    if 'quantize' in config['network'].keys():
        net = getattr(qlib, config['network']['quantize']['function'])(net, **config['network']['quantize']['kwargs'])

    net.eval()

    return net
