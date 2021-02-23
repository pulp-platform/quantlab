import torch
import horovod.torch as hvd


def train(logbook, train_l, net, loss_fn, opt, ctrls):
    """Run one epoch of the training experiment."""
    net.train()
    for c in ctrls:
        c.step_pre_training(logbook.i_epoch, opt)
    hvd.broadcast_parameters(net.state_dict(), root_rank=logbook.sw_cfg['master_rank'])

    logbook.meter.reset()
    for i_batch, data in enumerate(train_l):

        opt.zero_grad()
        # load data to device
        inputs, gt_labels = data
        inputs            = inputs.to(logbook.hw_cfg['device'])
        gt_labels         = gt_labels.to(logbook.hw_cfg['device'])
        # forprop
        pr_outs           = net(inputs)
        loss              = loss_fn(pr_outs, gt_labels)
        # backprop
        loss.backward()
        opt.step()

        # update statistics
        logbook.meter.update(pr_outs, gt_labels, loss)

        if logbook.verbose:
            print('Training\t [{:>4}/{:>4}]'.format(logbook.i_epoch+1, logbook.config['experiment']['n_epochs']), end='')
            print(' | Batch [{:>5}/{:>5}]'.format(i_batch+1, len(train_l)), end='')
            print(' | Loss: {:6.3f} - Metric: {:6.2f}'.format(logbook.meter.avg_loss, logbook.meter.avg_metric))

    # log statistics to file
    stats = {
        'train_loss':   logbook.meter.avg_loss,
        'train_metric': logbook.meter.avg_metric
    }
    if logbook.is_master:
        for k, v in stats.items():
            logbook.writer.add_scalar(k, v, global_step=logbook.i_epoch)
        logbook.writer.add_scalar('learning_rate', opt.param_groups[0]['lr'], global_step=logbook.i_epoch)

    return stats


def validate(logbook, valid_l, net, loss_fn, ctrls):
    """Run a validation epoch."""
    net.eval()
    for c in ctrls:
        c.step_pre_validation(logbook.i_epoch)

    with torch.no_grad():
        logbook.meter.reset()
        for i_batch, data in enumerate(valid_l):

            # load data to device
            inputs, gt_labels = data
            inputs            = inputs.to(logbook.hw_cfg['device'])
            gt_labels         = gt_labels.to(logbook.hw_cfg['device'])
            # forprop
            pr_outs           = net(inputs)
            loss              = loss_fn(pr_outs, gt_labels)
            # update statistics
            logbook.meter.update(pr_outs, gt_labels, loss)

            if logbook.verbose:
                print('Validation\t [{:>4}/{:>4}]'.format(logbook.i_epoch+1, logbook.config['experiment']['n_epochs']), end='')
                print(' | Batch [{:>5}/{:>5}]'.format(i_batch+1, len(valid_l)), end='')
                print(' | Loss: {:6.3f} - Metric: {:6.2f}'.format(logbook.meter.avg_loss, logbook.meter.avg_metric))

    # log statistics to file
    stats = {
        'valid_loss':   logbook.meter.avg_loss,
        'valid_metric': logbook.meter.avg_metric
    }
    if logbook.is_master:
        for k, v in stats.items():
            logbook.writer.add_scalar(k, v, global_step=logbook.i_epoch)

    return stats
