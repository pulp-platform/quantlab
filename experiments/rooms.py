from progress.bar import FillingSquaresBar
import torch


def train(logbook, train_l, net, device, loss_fn, opt):
    """Run one epoch of the training experiment."""
    logbook.meter.reset()
    bar = FillingSquaresBar('Training \t', max=len(train_l))
    for i_batch, data in enumerate(train_l):
        # load data onto device
        inputs, gt_labels = data
        inputs            = inputs.to(device)
        gt_labels         = gt_labels.to(device)
        # forprop
        pr_outs           = net(inputs)
        loss              = loss_fn(pr_outs, gt_labels)
        # backprop
        opt.zero_grad()
        loss.backward()
        opt.step()
        # update statistics
        logbook.meter.update(pr_outs, gt_labels, loss.item())#, track_metric=logbook.track_metric)
        bar.suffix = '[Epoch: {epoch:4d}] | [{batch:5d}/{num_batches:5d}]'.format(epoch=logbook.i_epoch, batch=i_batch+1, num_batches=len(train_l)) + logbook.meter.bar()
        bar.next()
    bar.finish()
    stats = {
        'train_loss':   logbook.meter.avg_loss,
        'train_metric': logbook.meter.avg_metric
    }
    for k, v in stats.items():
        if v:
            logbook.writer.add_scalar(k, v, global_step=logbook.i_epoch)
    logbook.writer.add_scalar('learning_rate', opt.param_groups[0]['lr'], global_step=logbook.i_epoch)
    return stats


def validate(logbook, valid_l, net, device, loss_fn):
    """Run a validation epoch."""
    logbook.meter.reset()
    bar = FillingSquaresBar('Validation \t', max=len(valid_l))
    with torch.no_grad():
        for i_batch, data in enumerate(valid_l):
            # load data onto device
            inputs, gt_labels     = data
            inputs                = inputs.to(device)
            gt_labels             = gt_labels.to(device)
            # forprop
            pr_outs               = net(inputs)
            loss                  = loss_fn(pr_outs, gt_labels)
            # update statistics
            logbook.meter.update(pr_outs, gt_labels, loss.item())#, track_metric=True)
            bar.suffix = '[Epoch: {epoch:4d}][{batch:5d}/{num_batches:5d}]'.format(epoch=logbook.i_epoch, batch=i_batch+1, num_batches=len(valid_l))
            bar.suffix = bar.suffix + logbook.meter.bar()
            bar.next()
    bar.finish()
    stats = {
        'valid_loss':   logbook.meter.avg_loss,
        'valid_metric': logbook.meter.avg_metric
    }
    for k, v in stats.items():
        if v:
            logbook.writer.add_scalar(k, v, global_step=logbook.i_epoch)
    return stats
