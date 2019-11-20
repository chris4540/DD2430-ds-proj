import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class ContrastiveLoss(_Loss):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1
    if samples are from the same class and label == 0 otherwise

    Notes:
        This is the implementation of eqn 1 in the paper
    """

    def __init__(self, margin, average=False):
        """
        Args:
            margin (float):
            average (bool): If taking average over batch, sum over batch otherwise
        """
        super().__init__()
        self.margin = margin
        # self.eps = 1e-9
        self.average = average

    def forward(self, input_, target):
        """
        Optimize later if performace is bad in computational graph using if
        """
        out1, out2 = input_

        # The squared distance
        dist_sq = torch.norm((out1 - out2), p=2, dim=0)  # The batch dim is 0

        if target == 1:
            losses = dist_sq
        else:
            losses = F.relu(self.margin**2 - dist_sq)

        if self.average:
            ret = losses.mean()
        else:
            ret = losses.sum()

        return ret
