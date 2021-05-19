# -*- coding: utf-8 -*-
from manager.platform import PlatformManager
from manager.logbook  import Logbook


def delete(args):

    platform = PlatformManager()
    platform.startup(horovod=args.horovod)

    logbook = Logbook(args.problem, args.topology)

    logbook.boot_logs_manager(exp_id=args.exp_id)
    logbook.logs_manager.destroy_exp_folder()
