import torch
import torch.nn as nn
import torch.optim as optim
import importlib

import utils.lr_schedulers as lr_schedulers


def get_data(logbook):
    """Return data for the experiment."""
    # create data sets
    train_set, valid_set = logbook.lib.load_data_sets(logbook)
    # is cross-validation experiment?
    if logbook.config['experiment']['n_folds'] > 1:
        import random
        import itertools
        indices = list(range(len(train_set)))
        # make data set random split consistent (to prevent that instance from training set filter into validation set)
        random.seed(logbook.seed)
        random.shuffle(indices)
        folds_indices = []
        for k in range(logbook.config['experiment']['n_folds']):
            folds_indices.append(indices[k::logbook.config['experiment']['n_folds']])
        train_fold_indices = list(itertools.chain(*[folds_indices[i] for i in range(len(folds_indices)) if i != logbook.i_fold]))
        valid_fold_indices = folds_indices[logbook.i_fold]
        valid_set = torch.utils.data.Subset(train_set, valid_fold_indices)
        train_set = torch.utils.data.Subset(train_set, train_fold_indices)  # overwriting `train_set` must be done in right order!
    # wrap data sets into loaders
    n_workers = logbook.config['data']['n_workers']
    bs_train  = logbook.config['data']['bs_train']
    bs_valid  = logbook.config['data']['bs_valid']
    if hasattr(train_set, 'collate_fn'):  # if one data set needs `collate`, all the data sets should need it
        train_l = torch.utils.data.DataLoader(train_set, batch_size=bs_train, shuffle=True,  num_workers=n_workers, collate_fn=train_set.collate_fn)
        valid_l = torch.utils.data.DataLoader(valid_set, batch_size=bs_valid, shuffle=False, num_workers=n_workers, collate_fn=valid_set.collate_fn)
    else:
        train_l = torch.utils.data.DataLoader(train_set, batch_size=bs_train, shuffle=True,  num_workers=n_workers)
        valid_l = torch.utils.data.DataLoader(valid_set, batch_size=bs_valid, shuffle=False, num_workers=n_workers)
    return train_l, valid_l


def get_network(logbook):
    """Return a network for the experiment and the loss function for training."""
    # create the network
    net = getattr(logbook.lib, logbook.config['network']['class'])(**logbook.config['network']['params'])
    # quantize (if specified)
    if logbook.config['network']['quantize'] is not None:
        quant_convert = getattr(logbook.lib, logbook.config['network']['quantize']['routine'])
        net = quant_convert(logbook, net)
    # load checkpoint state or pretrained network
    if logbook.ckpt:
        net.load_state_dict(logbook.ckpt['network'])
    # move to proper device and (if possible) parallelize
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
    net = net.to(device)
    if torch.cuda.device_count() > 1:
        net_maybe_par = nn.DataParallel(net)
    else:
        net_maybe_par = net
    return net, net_maybe_par, device


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
    opt_choice      = {**optim.__dict__}
    opt             = opt_choice[logbook.config['training']['optimizer']['class']](net.parameters(), **logbook.config['training']['optimizer']['params'])
    if logbook.ckpt:
        opt.load_state_dict(logbook.ckpt['training']['optimizer'])
    # learning rate scheduler
    lr_sched_choice = {**optim.lr_scheduler.__dict__, **lr_schedulers.__dict__}
    lr_sched        = lr_sched_choice[logbook.config['training']['lr_scheduler']['class']](opt, **logbook.config['training']['lr_scheduler']['params'])
    if logbook.ckpt:
        lr_sched.load_state_dict(logbook.ckpt['training']['lr_scheduler'])
    # quantization controllers (if specified)
    if logbook.config['training']['quantize']:
        f_controls = getattr(logbook.lib, logbook.config['training']['quantize']['routine'])
        ctrls = f_controls(logbook, net)
    else:
        ctrls = []
    return loss_fn, opt, lr_sched, ctrls
