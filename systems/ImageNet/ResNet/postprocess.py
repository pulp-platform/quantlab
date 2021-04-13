import torch


def postprocess_pr(pr_outs):
    _, topk = torch.topk(pr_outs.detach().cpu(), 5, dim=1)
    return [[a.item() for a in p] for p in topk]


def postprocess_gt(gt_labels):
    return [l.item() for l in gt_labels.detach().cpu()]
