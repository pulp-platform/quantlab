# 
# train.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# Georg Rutishauser <georgr@iis.ee.ethz.ch>
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

import argparse
import torch

from manager.platform   import PlatformManager
from manager.logbook    import Logbook
from manager.assistants import DataAssistant
from manager.assistants import NetworkAssistant
from manager.assistants import TrainingAssistant
from manager.assistants import MeterAssistant


def train(args: argparse.Namespace):
    """Train a DNN or (possibly) a QNN.

    This function implements QuantLab's training flow. The training
    flow applies the mini-batch stochastic gradient descent algorithm,
    or a variant of its, to optimise a target (possibly quantized)
    deep neural network. To corroborate the statistical reliability of
    experimental results, this flow supports :math:`K`-fold
    cross-validation (CV).

    At a high level, this function is structured in two main blocks:
    the *flow preamble* and the *training loop*.

    1. **Flow preamble**.
       The purpose of this part is setting up the bookkeeping
       infrastructure and preparing the software abstractions
       required to train the target DNN system. More in detail, it
       does the following:

       * import the software abstractions (classes, functions)
         required to assemble the learning system from the
         :ref:`systems <systems-package>` package;
       * load the parameters required to instantiate the software
         components of the learning system from the experimental
         unit's private logs folder, that is stored on disk;
       * parse and pass these two pieces of information to
         *assistants*, builder objects that are in charge of
         instantiating the software components of the learning
         system on-demand.

    2. **Training loop**.
       The purpose of this part is performing :math:`K` independent
       *training runs*, one for each fold specified by the CV setup.
       The runs are executed and numbered sequentially starting from
       zero (i.e., QuantLab does not support parallel execution of
       CV folds). Each training run consists of:

       * a *fold preamble*, during which the fold-specific logging
         structure is set up and the software components of the
         learning system are created by the builder objects; these
         components are re-instantiated from scratch (but according
         to the same specified configuration) at the beginning of
         each fold;
       * a *loop over epochs*; during each epoch, the flow will
         perform a loop over batches of training data points while
         optimising (i.e., training) the learning system, then
         perform a loop over batches of validation data points (no
         optimisation is performed during this phase), and, if
         necessary, store a checkpoint of the system's state; during
         the loop, statistics are collected.

    This function implements a checkpointing system to recover from
    unexpected interruptions of the training flow. In case of crashed
    or interrupted training runs, the flow will attempt to resume the
    loop from the most recent checkpoint of the run that was being
    performed when the flow was interrupted. This recovery is
    performed in two steps:

    * at the end of the *preamble*, the flow inspects the logs folder
      looking for the fold logs folder having the largest ID (recall
      that this flow carries out different CV folds sequentially); the
      experiment is resumed from the corresponding training run;
    * during the *fold preamble* of the resumed training run, after
      creating the software components of the learning system, the
      flow inspects the fold's checkpoint folder looking for the most
      recently created checkpoint file; the state of all the software
      components is restored, and the loop can continue from the
      same state it was in before the interruption.
    """

    platform = PlatformManager()
    platform.startup(horovod=args.horovod)

    # === FLOW: START ===

    # === FLOW: PREAMBLE ===

    # import the libraries required to assemble the ML system
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
    traindataassistant = DataAssistant('train')
    traindataassistant.recv_datamessage(logbook.send_datamessage('train'))
    validdataassistant = DataAssistant('valid')
    validdataassistant.recv_datamessage(logbook.send_datamessage('valid'))
    # network
    networkassistant = NetworkAssistant()
    networkassistant.recv_networkmessage(logbook.send_networkmessage())
    # training
    trainingassistant = TrainingAssistant()
    trainingassistant.recv_trainingmessage(logbook.send_trainingmessage())
    # meters
    trainmeterassistant = MeterAssistant('train')
    trainmeterassistant.recv_metermessage(logbook.send_metermessage('train'))
    validmeterassistant = MeterAssistant('valid')
    validmeterassistant.recv_metermessage(logbook.send_metermessage('valid'))

    # determine the status of cross-validation
    # [recovery] master-workers synchronisation point: find the fold ID by inspecting the experimental unit's logs folder
    if (not platform.is_horovod_run) or platform.is_master:
        logbook.logs_manager.discover_fold_id()
        start_fold_id = logbook.logs_manager.fold_id
    if platform.is_horovod_run:
        start_fold_id = platform.hvd.broadcast_object(start_fold_id, root_rank=platform.master_rank, name='start_fold_id')

    # === LOOP ON FOLDS: START ===

    # single cycle over CV folds (main loop of the experimental run)
    for fold_id in range(start_fold_id, logbook.n_folds):

        # === FOLD: START ===

        # === FOLD: PREAMBLE ===

        # master-only point: prepare fold logs folders
        if (not platform.is_horovod_run) or platform.is_master:
            logbook.logs_manager.setup_fold_logs(fold_id=fold_id)

        # prepare the system components for the current fold
        # data
        train_loader = traindataassistant.prepare(platform, fold_id)
        valid_loader = validdataassistant.prepare(platform, fold_id)
        # network
        net          = networkassistant.prepare(platform, fold_id)
        # training
        loss_fn      = trainingassistant.prepare_loss(net)
        gd           = trainingassistant.prepare_gd(platform, net)
        qnt_ctrls    = trainingassistant.prepare_qnt_ctrls(net)
        # meters
        train_meter  = trainmeterassistant.prepare(platform, len(train_loader), net, gd.opt)
        valid_meter  = validmeterassistant.prepare(platform, len(valid_loader), net)

        # [recovery] master-workers synchronisation point: load the latest checkpoint from disk
        if (not platform.is_horovod_run) or platform.is_master:
            start_epoch_id = logbook.logs_manager.load_checkpoint(platform, net, gd.opt, gd.lr_sched, qnt_ctrls, train_meter=train_meter, valid_meter=valid_meter)
        if platform.is_horovod_run:
            start_epoch_id = platform.hvd.broadcast_object(start_epoch_id, root_rank=platform.master_rank, name='start_epoch_id')
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
            train_meter.best_loss = platform.hvd.broadcast_object(train_meter.best_loss, root_rank=platform.master_rank, name='train_meter_best_loss')
            valid_meter.best_loss = platform.hvd.broadcast_object(valid_meter.best_loss, root_rank=platform.master_rank, name='valid_meter_best_loss')

        # if no checkpoint has been found, the epoch ID is set to -1
        # [recovery] if a checkpoint has been found, its epoch ID marks a completed epoch; the training should resume from the following epoch
        start_epoch_id += 1

        # === LOOP ON EPOCHS: START ===

        # master-only point: writer stubs are resolved to real TensorBoard writers
        # [recovery] the collected statistics about epochs and iterations carried out after the last stored checkpoint are erased
        if (not platform.is_horovod_run) or platform.is_master:
            logbook.logs_manager.create_writers(start_epoch_id=start_epoch_id, n_batches_train=len(train_loader), n_batches_valid=len(valid_loader))

        # cycle over epochs (one loop for each fold)
        for epoch_id in range(start_epoch_id, logbook.n_epochs):

            # === EPOCH: START ===

            # === TRAINING STEP: START ===
            net.train()

            # master-workers synchronisation point: quantization controllers might change the network's quantization parameters stochastically
            if (not platform.is_horovod_run) or platform.is_master:
                for c in qnt_ctrls:
                    c.step_pre_training_epoch(epoch_id)
            if platform.is_horovod_run:
                platform.hvd.broadcast_parameters(net.state_dict(), root_rank=platform.master_rank)

            # cycle over batches of training data (one loop for each epoch)
            for batch_id, (x, ygt) in enumerate(train_loader):

                # master-workers synchronisation point: quantization controllers might change the network's quantization parameters stochastically
                # TODO: in multi-process runs, synchronising processes at each step might be too costly
                if (not platform.is_horovod_run) or platform.is_master:
                    for c in qnt_ctrls:
                        c.step_pre_training_batch()
                if platform.is_horovod_run:
                    platform.hvd.broadcast_parameters(net.state_dict(), root_rank=platform.master_rank)

                # event: forward pass is beginning
                train_meter.step(epoch_id, batch_id)
                train_meter.start_observing()
                train_meter.tic()

                # processing (forward pass)
                x   = x.to(platform.device)
                ypr = net(x)

                # loss evaluation
                ygt  = ygt.to(platform.device)
                loss = loss_fn(ypr, ygt)

                # event: forward pass has ended; backward pass is beginning
                train_meter.update(ygt, ypr, loss)

                # training (backward pass)
                gd.opt.zero_grad()  # clear gradients
                loss.backward()     # gradient computation
                gd.opt.step()       # gradient descent

                # event: backward pass has ended
                train_meter.toc(ygt)
                train_meter.stop_observing()

            # === TRAINING STEP: END ===
            train_meter.check_improvement()

            # === VALIDATION STEP: START ===
            net.eval()

            # master-workers synchronisation point: quantization controllers might change the network's quantization parameters stochastically
            if (not platform.is_horovod_run) or platform.is_master:
                for c in qnt_ctrls:
                    c.step_pre_validation_epoch(epoch_id)
            if platform.is_horovod_run:
                platform.hvd.broadcast_parameters(net.state_dict(), root_rank=platform.master_rank)

            with torch.no_grad():  # no optimisation happens at validation time

                # cycle over batches of validation data (one loop for each epoch)
                for batch_id, (x, ygt) in enumerate(valid_loader):

                    # event: forward pass is beginning
                    valid_meter.step(epoch_id, batch_id)
                    valid_meter.start_observing()
                    valid_meter.tic()

                    # processing (forward pass)
                    x   = x.to(platform.device)
                    ypr = net(x)

                    # loss evaluation
                    ygt = ygt.to(platform.device)
                    loss = loss_fn(ypr, ygt)

                    # event: forward pass has ended
                    valid_meter.update(ygt, ypr, loss)
                    valid_meter.toc(ygt)
                    valid_meter.stop_observing()

            # === VALIDATION STEP: END ===
            valid_meter.check_improvement()

            # === EPOCH EPILOGUE ===

            # (possibly) change learning rate
            if gd.lr_sched is not None:
                gd.lr_sched.step()

            # has the target metric improved during the current epoch?
            if logbook.target_loss == 'train':
                is_best = train_meter.is_best
            elif logbook.target_loss == 'valid':
                is_best = valid_meter.is_best

            # master-only point: store checkpoint to disk if this is a checkpoint epoch or the target metric has improved during the current epoch
            if (not platform.is_horovod_run) or platform.is_master:
                if epoch_id % logbook.config['experiment']['ckpt_period'] == 0:  # checkpoint epoch; note that the first epoch is always a checkpoint epoch
                    logbook.logs_manager.store_checkpoint(epoch_id, net, gd.opt, gd.lr_sched, qnt_ctrls, train_meter, valid_meter)
                if epoch_id == logbook.n_epochs - 1:
                    logbook.logs_manager.store_checkpoint(epoch_id, net, gd.opt, gd.lr_sched, qnt_ctrls, train_meter, valid_meter)  # this is the last epoch
                if is_best:  # the target metric has improved during this epoch
                    logbook.logs_manager.store_checkpoint(epoch_id, net, gd.opt, gd.lr_sched, qnt_ctrls, train_meter, valid_meter, is_best=is_best)

            # === EPOCH: END ===

        # === LOOP ON EPOCHS: END ===

        # master-only point: when a ``SummaryWriter`` is built it is bound to a fold directory, so when a fold is completed it's time to destroy it
        if (not platform.is_horovod_run) or platform.is_master:
            logbook.logs_manager.destroy_writers()

        # reset starting epoch
        start_epoch_id = 0

        # === FOLD: END ===

    # === LOOP ON FOLDS: END ===

    # === FLOW: END ===
