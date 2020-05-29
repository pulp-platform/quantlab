import math


class Meter(object):
    def __init__(self, pp_pr, pp_gt):
        self.n_tracked    = None
        self.loss         = None
        self.avg_loss     = None
        # main metric is Top1 error
        self.pp_pr        = pp_pr
        self.pp_gt        = pp_gt
        self.start_metric = 100.
        self.topk         = (1, 5)
        self.correct      = None
        self.avg_error    = None
        self.avg_metric   = None
        self.reset()

    def reset(self):
        self.n_tracked = 0
        self.loss      = 0.
        self.avg_loss  = 0.
        self.correct   = [0 for k in self.topk]
        self.avg_error = [self.start_metric for k in self.topk]

    def update(self, pr_outs, gt_labels, loss):#, track_metric=False):
        gt_labels = self.pp_gt(gt_labels)
        batch_size      = len(gt_labels)
        self.n_tracked += batch_size
        # update loss
        self.loss      += loss * batch_size
        self.avg_loss   = self.loss / self.n_tracked
        if True:#track_metric:
            # update main metric
            pr_labels = self.pp_pr(pr_outs)
            assert len(pr_labels) == len(gt_labels), 'Number of predictions and number of ground truths do not match!'
            for i in range(len(pr_labels)):
                for n, k in enumerate(self.topk):
                    self.correct[n] += gt_labels[i] in pr_labels[i][0:k]
            self.avg_error = [100. * (1. - c / self.n_tracked) for c in self.correct]
            self.avg_metric = self.avg_error[0]

    def is_better(self, current_metric, best_metric):
        # compare Top1 errors
        return current_metric < best_metric

    def bar(self):
        return '| Loss: {loss:8.5f} | Top1: {top1:6.2f}%% | Top5: {top5:6.2f}%%'.format(
                loss=self.avg_loss,
                top1=100. - self.avg_error[0],
                top5=100. - self.avg_error[1])
