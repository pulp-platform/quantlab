import torch
import torch.nn as nn
import torch.optim as optim

import horovod.torch as hvd

import utils.lr_schedulers as lr_schedulers


def get_data(logbook):
    """Return data for the experiment."""

    # create data sets
    train_set, valid_set = logbook.lib.load_data_sets(logbook)
    # is cross-validation experiment?
    if logbook.config['experiment']['n_folds'] > 1:
        import itertools
        torch.manual_seed(logbook.config['experiment']['seed'])  # make data set random split consistent
        indices = torch.randperm(len(train_set)).tolist()
        folds_indices = []
        for k in range(logbook.config['experiment']['n_folds']):
            folds_indices.append(indices[k::logbook.config['experiment']['n_folds']])
        train_fold_indices = list(itertools.chain(*[folds_indices[i] for i in range(len(folds_indices)) if i != logbook.i_fold]))
        valid_fold_indices = folds_indices[logbook.i_fold]
        valid_set = torch.utils.data.Subset(train_set, valid_fold_indices)
        train_set = torch.utils.data.Subset(train_set, train_fold_indices)  # overwriting `train_set` must be done in right order!

    # create samplers (maybe distributed)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=logbook.sw_cfg['global_size'], rank=logbook.sw_cfg['global_rank'])
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_set, num_replicas=logbook.sw_cfg['global_size'], rank=logbook.sw_cfg['global_rank'])

    # wrap data sets into loaders
    bs_train = logbook.config['data']['bs_train']
    bs_valid = logbook.config['data']['bs_valid']
    kwargs = {'num_workers': logbook.hw_cfg['n_cpus'] // logbook.sw_cfg['local_size'], 'pin_memory': True} if logbook.hw_cfg['n_gpus'] else {'num_workers': 1}
    if hasattr(train_set, 'collate_fn'):  # if one data set needs `collate`, all the data sets should need it
        train_l = torch.utils.data.DataLoader(train_set, batch_size=bs_train, sampler=train_sampler, collate_fn=train_set.collate_fn, **kwargs)
        valid_l = torch.utils.data.DataLoader(valid_set, batch_size=bs_valid, sampler=valid_sampler, collate_fn=valid_set.collate_fn, **kwargs)
    else:
        train_l = torch.utils.data.DataLoader(train_set, batch_size=bs_train, sampler=train_sampler, **kwargs)
        valid_l = torch.utils.data.DataLoader(valid_set, batch_size=bs_valid, sampler=valid_sampler, **kwargs)

    return train_l, valid_l


def get_network(logbook):
    """Return a network for the experiment and the loss function for training."""

    # create the network
    net = getattr(logbook.lib, logbook.config['network']['class'])(**logbook.config['network']['params'])

    # quantize (if specified)
    if logbook.config['network']['quantize'] is not None:
        quant_convert = getattr(logbook.lib, logbook.config['network']['quantize']['routine'])
        net = quant_convert(logbook.config['network']['quantize'], net)

    # move to proper device
    net = net.to(logbook.hw_cfg['device'])

    return net


def get_training(logbook, net):
    """Return a training procedure for the experiment."""

    # loss function
    loss_fn_choice = {**nn.__dict__, **logbook.lib.__dict__}
    loss_fn_class  = loss_fn_choice[logbook.config['training']['loss_function']['class']]
    if 'net' in loss_fn_class.__init__.__code__.co_varnames:
        loss_fn = loss_fn_class(net, **logbook.config['training']['loss_function']['params'])
    else:
        loss_fn = loss_fn_class(**logbook.config['training']['loss_function']['params'])

    # optimization algorithm
    opt_choice = {**optim.__dict__}
    logbook.config['training']['optimizer']['params']['lr'] *= logbook.sw_cfg['global_size']  # adjust learning rate
    opt        = opt_choice[logbook.config['training']['optimizer']['class']](net.parameters(), **logbook.config['training']['optimizer']['params'])
    opt        = hvd.DistributedOptimizer(opt, named_parameters=net.named_parameters())

    # learning rate scheduler
    lr_sched_choice = {**optim.lr_scheduler.__dict__, **lr_schedulers.__dict__}
    lr_sched        = lr_sched_choice[logbook.config['training']['lr_scheduler']['class']](opt, **logbook.config['training']['lr_scheduler']['params'])

    # quantization controllers (if specified)
    if logbook.config['training']['quantize']:
        quant_controls = getattr(logbook.lib, logbook.config['training']['quantize']['routine'])
        ctrls = quant_controls(logbook.config['training']['quantize'], net)
    else:
        ctrls = []

    return loss_fn, opt, lr_sched, ctrls
