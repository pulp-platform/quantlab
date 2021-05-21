# -*- coding: utf-8 -*-
from manager.platform import PlatformManager
from manager.logbook  import Logbook


def configure(args):

    platform = PlatformManager()
    platform.startup(horovod=args.horovod)

    logbook = Logbook(args.problem, args.topology)

    logbook.create_config(args.target_loss, args.ckpt_period,
                          args.n_folds, args.cv_seed,
                          args.fix_sampler, args.sampler_seed,
                          args.fix_network, args.network_seed)

    logbook.boot_logs_manager()
    logbook.logs_manager.create_exp_folder()
    logbook.logs_manager.store_config(logbook.config)
