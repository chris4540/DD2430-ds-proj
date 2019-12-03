"""
Customized metrics

https://github.com/pytorch/ignite/blob/master/ignite/engine/__init__.py
https://github.com/pytorch/ignite/blob/master/ignite/metrics/mean_squared_error.py
"""
import torch
from ignite.metrics import Accuracy


class SiameseNetSimilarityAccuracy(Accuracy):
    """
    Calculate the similarity of a siamese network

    Example:
        eval = create_embedding_engine(..., )
    """

    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def update(self, output):
        # calculate output
        y_pred, y = output

        out1, out2 = y_pred
        _, _, is_similar = y

        # calculate L2 vector norm over the embedding dim
        dist = torch.norm((out1 - out2), p=2, dim=1)
        # The squared distance
        dist_sq = dist.pow(2)

        # dist_sq < margin**2 => similar => 1
        pred = (dist_sq < self.margin**2).int()
        super().update((pred, is_similar))
