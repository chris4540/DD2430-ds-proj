# """
# Customized metrics

# https://github.com/pytorch/ignite/blob/master/ignite/engine/__init__.py
# https://github.com/pytorch/ignite/blob/master/ignite/metrics/mean_squared_error.py
# """
# import torch
# from ignite.metrics import Accuracy
# import torch.nn.functional as F


# class SiameseNetSimilarityAccuracy(Accuracy):
#     """
#     Calculate the similarity of a siamese network

#     Example:
#         eval = create_embedding_engine(..., )
#     """

#     def __init__(self, margin, l2_normalize=False):
#         super().__init__()
#         self.margin = margin
#         self.l2_normalize = l2_normalize

#     def update(self, output):
#         # calculate output
#         y_pred, y = output

#         out1, out2 = y_pred
#         if self.l2_normalize:
#             out1 = F.normalize(out1, p=2, dim=1)
#             out2 = F.normalize(out2, p=2, dim=1)

#         _, _, is_similar = y

#         # calculate L2 vector norm over the embedding dim
#         dist = torch.norm((out1 - out2), p=2, dim=1)
#         # The squared distance
#         dist_sq = dist.pow(2)

#         # dist_sq < margin**2 => similar => 1
#         pred = (dist_sq < self.margin**2).int()
#         super().update((pred, is_similar))
