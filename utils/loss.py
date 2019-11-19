import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super().__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        
        targ_f = target.float()
        losses = .5 * (targ_f * distances +
            (1. - targ_f) * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))

        ret = losses.mean() if size_average else losses.sum()
        return ret
