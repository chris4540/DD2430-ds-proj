"""
Test loss function
"""
import torch
from utils.loss import ContrastiveLoss


def test_contrastive_loss():
    batch_size = 100
    emb_size = 512
    emb_vec1 = torch.randn((batch_size, emb_size))
    emb_vec2 = torch.randn((batch_size, emb_size))
    targets = torch.randint(2, size=(100, 1))

    loss_fn = ContrastiveLoss(margin=1.0)

    loss = loss_fn((emb_vec1, emb_vec2), targets)

    assert len(loss.shape) == 0
    assert loss > 0
