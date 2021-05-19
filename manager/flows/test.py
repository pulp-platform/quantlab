# -*- coding: utf-8 -*-
import torch

from manager.platform   import PlatformManager
from manager.logbook    import Logbook
from manager.assistants import DataAssistant
from manager.assistants import NetworkAssistant
from manager.assistants import TrainingAssistant
from manager.assistants import MeterAssistant


def test(args):

    platform = PlatformManager()
    platform.startup(horovod=args.horovod)

    # determine the libraries required to assemble the ML system
    logbook = Logbook(args.problem, args.topology)

    # master-only point: in multi-process runs, each process creates a logbook, but only one is privileged enough to interact with the disk
    if (not platform.is_horovod_run) or platform.is_master:
        logbook.open_logs(exp_id=args.exp_id)

    # master-workers synchronisation point: load configuration from disk
    if (not platform.is_horovod_run) or platform.is_master:
        logbook.load_config()
    if platform.is_horovod_run:
        logbook.config = platform.hvd.broadcast_object(logbook.config, root_rank=platform.master_rank, name='config')

    # prepare assistants
    # data
    dataassistant = DataAssistant()
    dataassistant.recv_datamessage(logbook.send_datamessage())
    # network
    networkassistant = NetworkAssistant()
    networkassistant.recv_networkmessage(logbook.send_networkmessage())
    # training
    trainingassistant = TrainingAssistant()
    trainingassistant.recv_trainingmessage(logbook.send_trainingmessage())
    # meters
    meterassistant = MeterAssistant()
    meterassistant.recv_metermessage(logbook.send_metermessage())

    # master-workers synchronisation point: set fold ID
    if (not platform.is_horovod_run) or platform.is_master:
        logbook.set_fold_id(fold_id=args.fold_id)
    if platform.is_horovod_run:
        logbook.fold_id = platform.hvd.broadcast_object(logbook.fold_id, root_rank=platform.master_rank, name='fold_id')

    # master-only point: prepare fold logs folders
    if (not platform.is_horovod_run) or platform.is_master:
        logbook.setup_fold_logs()

    # prepare the entities for the current fold
    # data
    train_loader, valid_loader = dataassistant.prepare(platform, logbook.fold_id)
    # network
    net                        = networkassistant.prepare(platform, logbook.fold_id)
    # training
    loss_fn, gd, qnt_ctrls     = trainingassistant.prepare(platform, net)
    # meters
    meter_train, meter_valid   = meterassistant.prepare(platform, logbook.logs_manager, len(train_loader), len(valid_loader), net, gd.opt)

    logbook.set_n_epochs()
    # master-workers synchronisation point: load the desired checkpoint from the fold's logs folder
    if (not platform.is_horovod_run) or platform.is_master:
        logbook.epoch_id = logbook.logs_manager.load_checkpoint(net, gd.opt, gd.lr_sched, qnt_ctrls, meter_train, meter_valid, ckpt_id=args.ckpt_id)
    if platform.is_horovod_run:
        logbook.epoch_id = platform.hvd.broadcast_object(logbook.epoch_id, root_rank=platform.master_rank, name='epoch_id')
        platform.hvd.broadcast_parameters(net.state_dict(), root_rank=platform.master_rank)
        platform.hvd.broadcast_optimizer_state(gd.opt, root_rank=platform.master_rank)
        if gd.lr_sched is not None:
            sd_lr_sched = platform.hvd.broadcast_object(gd.lr_sched.state_dict(), root_rank=platform.master_rank, name='sd_lr_sched')
            if not platform.is_master:
                gd.lr_sched.load_state_dict(sd_lr_sched)
        for i, c in enumerate(qnt_ctrls):
            sd_c = platform.hvd.broadcast_object(c.state_dict(), root_rank=platform.master_rank, name='sd_c_{}'.format(i))
            if not platform.is_master:
                c.load_state_dict(sd_c)
        meter_train.best_loss = platform.hvd.broadcast_object(meter_train.best_loss, root_rank=platform.master_rank, name='meter_train_best_loss')
        meter_valid.best_loss = platform.hvd.broadcast_object(meter_valid.best_loss, root_rank=platform.master_rank, name='meter_valid_best_loss')

    # === MAIN TESTING LOOP ===
    net.eval()

    # master-workers synchronisation point: quantization controllers might change the network's quantization parameters stochastically
    if (not platform.is_horovod_run) or platform.is_master:
        for c in qnt_ctrls:
            c.step(logbook.epoch_id)
    if platform.is_horovod_run:
        platform.hvd.broadcast_parameters(net.state_dict(), root_rank=platform.master_rank)

    with torch.no_grad():

        for batch_id, (x, ygt) in enumerate(valid_loader):

            meter_valid.step(logbook.epoch_id, batch_id)
            meter_valid.start_observing()
            meter_valid.tic()

            # processing (forward pass)
            x = x.to(platform.device)
            ypr = net(x)

            # loss evaluation
            ygt = ygt.to(platform.device)
            loss = loss_fn(ypr, ygt)

            meter_valid.update(ygt, ypr, loss)
            meter_valid.toc(ygt)
            meter_valid.stop_observing()
