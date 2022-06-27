# 
# test.py
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
        logbook.boot_logs_manager(exp_id=args.exp_id)

    # master-workers synchronisation point: load configuration from disk
    if (not platform.is_horovod_run) or platform.is_master:
        logbook.config = logbook.logs_manager.load_config()
    if platform.is_horovod_run:
        logbook.config = platform.hvd.broadcast_object(logbook.config, root_rank=platform.master_rank, name='config')

    # prepare assistants
    # data
    testdataassistant = DataAssistant('valid')  # TODO: be sure that `load_data_set` has a valid test branch
    testdataassistant.recv_datamessage(logbook.send_datamessage('valid'))  # TODO: maybe the user would like to test images one at a time, so we should define a new `data` sub-section for test settings
    # network
    networkassistant = NetworkAssistant()
    networkassistant.recv_networkmessage(logbook.send_networkmessage())
    # training
    trainingassistant = TrainingAssistant()
    trainingassistant.recv_trainingmessage(logbook.send_trainingmessage())
    # meters
    testmeterassistant = MeterAssistant('valid')  # TODO: be sure that `MeterAssistant` can create proper test meter (e.g., no LR statistic, no profiling, ...
    testmeterassistant.recv_metermessage(logbook.send_metermessage('valid'))

    # master-workers synchronisation point: set fold ID to the one containing the required checkpoint
    if (not platform.is_horovod_run) or platform.is_master:
        logbook.logs_manager.set_fold_id(fold_id=args.fold_id)
    if platform.is_horovod_run:
        logbook.fold_id = platform.hvd.broadcast_object(logbook.logs_manager._fold_id, root_rank=platform.master_rank, name='fold_id')
    else:
        logbook.fold_id = logbook.logs_manager._fold_id

    # master-only point: prepare fold logs folders
    if (not platform.is_horovod_run) or platform.is_master:
        logbook.logs_manager.setup_fold_logs(fold_id=args.fold_id)

    # prepare the entities for the test
    # data
    test_loader = testdataassistant.prepare(platform, logbook.fold_id)
    # network
    net         = networkassistant.prepare(platform, logbook.fold_id)
    # training
    loss_fn     = trainingassistant.prepare_loss(net)
    gd          = trainingassistant.prepare_gd(platform, net)
    qnt_ctrls   = trainingassistant.prepare_qnt_ctrls(net)
    # meters
    test_meter  = testmeterassistant.prepare(platform, len(test_loader), net, gd.opt)

    # master-workers synchronisation point: load the desired checkpoint from the fold's logs folder
    if (not platform.is_horovod_run) or platform.is_master:
        logbook.epoch_id = logbook.logs_manager.load_checkpoint(platform, net, gd.opt, gd.lr_sched, qnt_ctrls, ckpt_id=args.ckpt_id)
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

    # === MAIN TESTING LOOP ===
    net.eval()
    # master-workers synchronisation point: quantization controllers might change the network's quantization parameters stochastically
    if (not platform.is_horovod_run) or platform.is_master:
        for c in qnt_ctrls:
            c.step_pre_training_epoch(logbook.epoch_id)
            c.step_pre_validation_epoch(logbook.epoch_id)
    if platform.is_horovod_run:
        platform.hvd.broadcast_parameters(net.state_dict(), root_rank=platform.master_rank)

    with torch.no_grad():

        for batch_id, (x, ygt) in enumerate(test_loader):

            test_meter.step(logbook.epoch_id, batch_id)
            test_meter.start_observing()
            test_meter.tic()

            # processing (forward pass)
            x = x.to(platform.device)
            ypr = net(x)

            # loss evaluation
            ygt = ygt.to(platform.device)
            loss = loss_fn(ypr, ygt)

            test_meter.update(ygt, ypr, loss)
            test_meter.toc(ygt)
            test_meter.stop_observing()

