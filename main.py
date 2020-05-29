import argparse

from experiments import Logbook
from experiments import get_data, get_network, get_training
from experiments import train, validate


# Command Line Interface
parser = argparse.ArgumentParser(description='QuantLab')
parser.add_argument('--problem',    help='ImageNet')
parser.add_argument('--topology',   help='Network topology')
parser.add_argument('--exp_id',     help='Experiment to launch/resume/test',      default=None)
parser.add_argument('--mode',       help='Mode: train/test',                      default='train')
parser.add_argument('--ckpt_every', help='Frequency of checkpoints (in epochs)',  default=10)
args = parser.parse_args()

# create/retrieve experiment logbook
logbook = Logbook(args.problem, args.topology, args.exp_id)

# run experiment
if args.mode == 'train':
    for i_fold in range(logbook.i_fold, logbook.config['experiment']['n_folds']):
        logbook.open_fold()
        # create data sets for current fold
        train_l, valid_l = get_data(logbook)
        # load last checkpoint for current fold
        logbook.load_checkpoint('last')
        net, net_maybe_par, device = get_network(logbook)
        loss_fn, opt, lr_sched, ctrls = get_training(logbook, net)
        # run/complete current fold
        logbook.init_measurements()
        for i_epoch in range(logbook.i_epoch, logbook.config['experiment']['n_epochs']):
            logbook.open_epoch()
            # train
            net.train()
            for c in ctrls:
                c.step_pre_training(logbook.i_epoch, opt)
            train_stats = train(logbook, train_l, net_maybe_par, device, loss_fn, opt)
            # validate
            net.eval()
            for c in ctrls:
                c.step_pre_validation(logbook.i_epoch)
            valid_stats = validate(logbook, valid_l, net, device, loss_fn)
            # (maybe) update learning rate
            stats = {**train_stats, **valid_stats}
            if 'metrics' in lr_sched.step.__code__.co_varnames:
                lr_sched_metric = stats[logbook.config['training']['lr_scheduler']['step_metric']]
                lr_sched.step(lr_sched_metric)
            else:
                lr_sched.step()
            # save model if checkpoint epoch and/or if update metric has improved
            is_ckpt_epoch = (logbook.i_epoch % args.ckpt_every) == 0
            is_best = logbook.is_better(stats)
            if is_ckpt_epoch or is_best:
                ckpt = {
                    'fold': {
                        'i_epoch': logbook.i_epoch,
                        'metrics': logbook.metrics
                    },
                    'network': net.state_dict(),
                    'training': {
                        # 'controllers':   thr.state_dict(),
                        'optimizer':    opt.state_dict(),
                        'lr_scheduler': lr_sched.state_dict()
                    }
                }
                if is_ckpt_epoch:
                    logbook.store_checkpoint(ckpt)
                if is_best:
                    logbook.store_checkpoint(ckpt, is_best=True)
            logbook.close_epoch()
        logbook.close_fold()
# elif args.mode == 'test':
#     # test
#     net.eval()
#     test_stats = test(logbook, net, device, loss_fn, test_l)
