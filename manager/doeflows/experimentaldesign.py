from collections import namedtuple
import pandas as pd
from enum import EnumMeta
import os
from openpyxl import load_workbook
import importlib
import json
import collections.abc

from typing import Tuple, List, Dict
from typing import Union
from collections.abc import Iterator


Configuration = namedtuple('Configuration', ['setup', 'patch'])


class ExperimentalDesign(object):
    """
    Attributes:
        _configs (List[Configuration]): the
    """
    def __init__(self, dofs: Tuple[EnumMeta]) -> None:
        self._base_cfg = None
        self._load_base_cfg()
        self._dofs     = dofs  # degrees of freedom
        self._configs  = list()  # List[Configuration]
        self._generate_configs()

    @property
    def base_cfg(self) -> Dict:
        return self._base_cfg

    @property
    def dofs(self) -> Tuple[EnumMeta]:
        return self._dofs

    def _load_base_cfg(self):
        """Generate a basic configuration dictionary that will be patched."""
        # with open(os.path.join(os.path.dirname(__file__), '.'.join([self.__class__.__name__, 'json']), 'r')) as fp:
        #     self._base_cfg = json.load(fp)
        raise NotImplementedError

    def _generate_configs(self):
        """Generate configurations and append them to the experiments list."""
        raise NotImplementedError

    @staticmethod
    def patch_config(config, patch):
        # https://stackoverflow.com/a/3233356
        for k, v in patch.items():
            if isinstance(v, collections.abc.Mapping):
                config[k] = ExperimentalDesign.patch_config(config.get(k, {}), v)
            else:
                config[k] = v
        return config

    def __iter__(self):
        return iter(self._configs)


class ExperimentalDesignLogger(object):

    def __init__(self,
                 problem: str,
                 topology: str,
                 exp_design: str) -> None:

        self._filename = os.path.join('systems', problem, topology, 'logs', '.'.join([exp_design, 'xlsx']))

        doe_module = importlib.import_module('.doe', package='.'.join(['systems', problem, topology]))
        self.ed    = getattr(doe_module, exp_design)()

        self._dofs = {dof.__name__: [(v.name, v.value) for v in dof] for dof in self.ed.dofs}
        self._flush_dofs()

        self._cfg_gen = iter(self.ed)  # iterator over configurations
        self._record  = list()

    @property
    def cfg_iterator(self) -> Iterator:
        return iter(self.ed)

    def _flush_dofs(self) -> None:

        with pd.ExcelWriter(self._filename, engine='openpyxl') as ew:

            for name, mapping in self._dofs.items():
                df = pd.DataFrame(data=mapping, columns=['name', 'value'])
                df.to_excel(ew, sheet_name=name, index=False)

    def update_record(self, eu_id: int, setup: Tuple[Union[None, int, float, str]]):
        self._record.append((eu_id, *setup))

    def flush_record(self):

        # Panda's `ExcelWriter` does not have a native "append sheet" mode.
        # I found a workaround here:
        #
        #   https://soulsinporto.medium.com/how-to-add-new-worksheets-to-excel-workbooks-with-pandas-47122704fb75
        #
        if os.path.exists(self._filename):
            workbook = load_workbook(self._filename)

        with pd.ExcelWriter(self._filename, engine='openpyxl') as ew:
            ew.book   = workbook
            ew.sheets = {worksheet.title: worksheet for worksheet in workbook.worksheets}

            df = pd.DataFrame(data=self._record, columns=['ID'] + list(self._dofs.keys()))
            df.to_excel(ew, sheet_name='ExperimentalUnits', index=False)
            ew.save()
