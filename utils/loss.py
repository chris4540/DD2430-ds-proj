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
        self.average = average

    def forward(self, input_, targets):
        """
        Optimize later if performace is bad in computational graph using if
        """
        # Unpack
        out1, out2 = input_
        c1, c2, target = targets

        # calculate L2 vector norm over the embedding dim
        dist = torch.norm((out1 - out2), p=2, dim=1)
        # The squared distance
        dist_sq = dist.pow(2)

        # Penalty for similar images
        sim_img_loss = dist_sq

        # Penalty for dissimilar images
        dissim_img_loss = F.relu(self.margin**2 - dist_sq)

        y = target.float()
        losses = y * sim_img_loss + (1. - y) * dissim_img_loss

        if self.average:
            ret = losses.mean()
        else:
            ret = losses.sum()

        return ret
