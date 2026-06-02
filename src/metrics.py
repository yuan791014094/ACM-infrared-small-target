"""Soft-IoU loss and IoU / nIoU evaluation metrics.

Soft-IoU (the paper's loss): differentiable IoU on sigmoid probabilities.
IoU:  aggregate intersection over aggregate union across the whole set.
nIoU (paper Eq. 1): mean of per-sample IoU,
    nIoU = (1/N) * sum_i TP[i] / (T[i] + P[i] - TP[i]).
"""

import torch
import torch.nn as nn


class SoftIoULoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        prob = prob.view(prob.size(0), -1)
        target = target.view(target.size(0), -1)
        inter = (prob * target).sum(dim=1)
        total = prob.sum(dim=1) + target.sum(dim=1)
        union = total - inter
        iou = (inter + self.smooth) / (union + self.smooth)
        return 1.0 - iou.mean()


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al. 2017) on logits, for the ACM++ hybrid loss.

    Targets the extreme foreground/background imbalance of tiny IR targets.
    """

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        ce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target, reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * ce
        return loss.mean()


class HybridLoss(nn.Module):
    """ACM++ improvement #3:  L = SoftIoU + lambda * Focal.

    Supports deep supervision: if `logits` is a list [main, aux, ...], the loss
    is computed on each and aux terms are down-weighted by `aux_weight`.
    """

    def __init__(self, focal_weight=1.0, aux_weight=0.4,
                 alpha=0.75, gamma=2.0, smooth=1.0):
        super().__init__()
        self.softiou = SoftIoULoss(smooth)
        self.focal = FocalLoss(alpha, gamma)
        self.focal_weight = focal_weight
        self.aux_weight = aux_weight

    def _single(self, logits, target):
        return self.softiou(logits, target) + self.focal_weight * self.focal(logits, target)

    def forward(self, logits, target):
        if isinstance(logits, (list, tuple)):
            main = self._single(logits[0], target)
            aux = sum(self._single(l, target) for l in logits[1:])
            return main + self.aux_weight * aux
        return self._single(logits, target)


class SegMetrics:
    """Accumulates statistics over a dataset to compute IoU and nIoU.

    Predictions are thresholded at `thresh` on the sigmoid probability.
    """

    def __init__(self, thresh=0.5):
        self.thresh = thresh
        self.reset()

    def reset(self):
        self.total_inter = 0.0
        self.total_union = 0.0
        self.niou_sum = 0.0
        self.n = 0

    @torch.no_grad()
    def update(self, logits, target):
        if isinstance(logits, (list, tuple)):
            logits = logits[0]  # metrics use the main head only
        pred = (torch.sigmoid(logits) > self.thresh).float()
        target = (target > 0.5).float()
        b = pred.size(0)
        p = pred.view(b, -1)
        t = target.view(b, -1)
        tp = (p * t).sum(dim=1)
        pp = p.sum(dim=1)
        tt = t.sum(dim=1)
        union = pp + tt - tp

        # aggregate IoU
        self.total_inter += tp.sum().item()
        self.total_union += union.sum().item()

        # per-sample nIoU (define empty/empty as IoU=1)
        per = torch.where(union > 0, tp / (union + 1e-12), torch.ones_like(tp))
        self.niou_sum += per.sum().item()
        self.n += b

    def compute(self):
        iou = self.total_inter / (self.total_union + 1e-12)
        niou = self.niou_sum / max(self.n, 1)
        return {"IoU": iou, "nIoU": niou}
