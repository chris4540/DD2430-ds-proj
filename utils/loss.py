import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class ContrastiveLoss(_Loss):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1
    if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin, size_average=True):
        super().__init__()
        self.margin = margin
        self.eps = 1e-9
        self.size_average = size_average

    def forward(self, input, target):
        output1, output2 = input
        distances = (output2 - output1).pow(2).sum(1)  # squared distances

        targ_f = target.float()
        losses = .5 * (targ_f * distances +
                       (1. - targ_f) * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))

        ret = losses.mean() if self.size_average else losses.sum()
        return ret
