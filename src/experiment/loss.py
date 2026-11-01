"""Binary path of uploaded utils.WeightedDiceBCE, without unused dependencies.

The equations, smoothing and coefficients are preserved. The Dice uses squared
denominators and is a training surrogate; reporting uses protocol_metrics.py.
"""
import torch
from torch import nn
from torch.nn import functional as F


class WeightedDiceBCE(nn.Module):
    def __init__(self, dice_weight=0.5, BCE_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.BCE_weight = BCE_weight

    def forward(self, probability, target):
        if probability.shape != target.shape or probability.ndim != 4 or probability.shape[1] != 1:
            raise ValueError('Binary LViT loss requires equal [B,1,H,W] tensors')
        p = probability.reshape(len(probability), -1)
        t = target.reshape(len(target), -1)
        # Equal foreground/background weights in the original WeightedDiceLoss.
        weighted_p, weighted_t = 0.5 * p, 0.5 * t
        intersection = (weighted_p * weighted_t).sum(-1)
        union = weighted_p.square().sum(-1) + weighted_t.square().sum(-1)
        dice_loss = (1 - (2 * intersection + 1e-5) / (union + 1e-5)).mean()
        pixel_bce = F.binary_cross_entropy(p, t, reduction='none')
        positive, negative = (t > 0.5).float(), (t < 0.5).float()
        # Class normalization is over the batch, matching the upstream expression.
        bce = (0.5 * positive * pixel_bce / (positive.sum().item() + 1e-12)
               + 0.5 * negative * pixel_bce / (negative.sum().item() + 1e-12)).sum()
        return self.dice_weight * dice_loss + self.BCE_weight * bce

