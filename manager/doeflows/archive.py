import os
import subprocess
import shutil

from .experimentaldesign import ExperimentalDesignLogger
import manager
import manager.logbook.logsmanager

from typing import Union


def archive_exp(path_archive: Union[str, os.PathLike], eu_id: int, save_storage: bool) -> None:

    dir_logs = os.path.dirname(path_archive)
    dir_exp = manager.logbook.logsmanager._FORMAT_EXP_DIR.format(eu_id)
    path_exp = os.path.join(dir_logs, dir_exp)
    file_tar = '.'.join([dir_exp, 'tar', 'gz'])
    path_tar = os.path.join(path_archive, file_tar)

    subprocess.run('tar -pczf {} -C {} {}'.format(path_tar, os.path.dirname(path_exp), os.path.basename(path_exp)), shell=True)

    if save_storage:
        shutil.rmtree(path_exp)


def archive(args):

    logger = ExperimentalDesignLogger(args.problem, args.topology, args.exp_design)
    logger.load_register()

    if logger.is_experiment_archivable():

        path_archive = logger.get_path_archive()

        for eu_status, eu_id, *eu_dofs_values in logger.eu_register:
            archive_exp(path_archive, eu_id, args.save_storage)

        logger.move_edl(path_archive)
        if args.save_storage:
            logger.delete_edl()

    else:
        print(manager.QUANTLAB_PREFIX + "Some experiments have not been executed to completion. Aborting archiviation flow.")
