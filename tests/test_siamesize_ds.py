import numpy as np
import pytest
from utils.datasets import DeepFashionDataset
from utils.datasets import Siamesize
from config.deep_fashion import root_dir
from config.fashion_mnist import FashionMNISTConfig
from utils.datasets import FashionMNIST

def test_siamesize_mnist_basic():
    cfg = FashionMNISTConfig

    ds_kwargs = {
        'root': cfg.root,
    }
    train_ds = FashionMNIST(train=True, download=True, **ds_kwargs)
    siamese_ds = Siamesize(train_ds)

    targets = list()
    for idx1 in range(5000):
        idx2, is_similar = siamese_ds._get_idx2_and_target(idx1)
        assert is_similar in [0, 1]
        assert idx1 != idx2
        # store the targets
        targets.append(is_similar)

    # empirically check probability of the target getting 0 or 1
    prob = np.mean(targets)
    assert 0.4 <= prob <= 0.6

def test_siamesize_train_ds_basic():
    """
    Basic testing
    """
    dataset = DeepFashionDataset(root_dir, 'train')
    siamese_ds = Siamesize(dataset)

    targets = list()
    for idx1 in range(5000):
        idx2, is_similar = siamese_ds._get_idx2_and_target(idx1)
        assert is_similar in [0, 1]
        assert idx1 != idx2
        # store the targets
        targets.append(is_similar)

    # empirically check probability of the target getting 0 or 1
    prob = np.mean(targets)
    assert 0.4 <= prob <= 0.6

def test_siamesize_train_ds():
    """
    One epoch testing
    """
    dataset = DeepFashionDataset(root_dir, 'train')
    siamese_ds = Siamesize(dataset)

    targets = list()
    for idx1 in range(len(siamese_ds)):
        idx2, is_similar = siamese_ds._get_idx2_and_target(idx1)
        # store the targets
        targets.append(is_similar)

    # empirically check probability of the target getting 0 or 1
    prob = np.mean(targets)
    assert 0.48 <= prob <= 0.52

def test_siamesize_val_ds():
    dataset = DeepFashionDataset(root_dir, 'val')
    siamese_ds = Siamesize(dataset)
    assert not siamese_ds.train

    # first round
    pairs = list()
    targets = list()
    for idx1 in range(len(siamese_ds)):
        idx2, is_similar = siamese_ds._get_idx2_and_target(idx1)
        assert is_similar in [0, 1]
        assert idx1 != idx2
        # store the targets
        pairs.append((idx1, idx2))
        targets.append(is_similar)

    prob = np.mean(targets)
    assert 0.48 <= prob <= 0.52

    # second round
    for idx1 in range(len(siamese_ds)):
        idx2, is_similar = siamese_ds._get_idx2_and_target(idx1)
        assert (idx1, idx2) in pairs


def test_siamesize_test_ds():
    dataset = DeepFashionDataset(root_dir, 'test')
    siamese_ds = Siamesize(dataset)
    assert not siamese_ds.train

    # first round
    pairs = list()
    targets = list()
    for idx1 in range(len(siamese_ds)):
        idx2, is_similar = siamese_ds._get_idx2_and_target(idx1)
        assert is_similar in [0, 1]
        assert idx1 != idx2
        # store the targets
        pairs.append((idx1, idx2))
        targets.append(is_similar)

    prob = np.mean(targets)
    assert 0.48 <= prob <= 0.52

    # second round
    for idx1 in range(len(siamese_ds)):
        idx2, is_similar = siamese_ds._get_idx2_and_target(idx1)
        assert (idx1, idx2) in pairs

@pytest.mark.timeout(200)
def test_siamesize_train_ds_epoch():
    n_epochs = 10
    dataset = DeepFashionDataset(root_dir, 'val')

    # make it trainable
    dataset.train = True
    siamese_ds = Siamesize(dataset)

    for epoch in range(n_epochs):
        for idx1 in range(len(siamese_ds)):
            idx2, is_similar = siamese_ds._get_idx2_and_target(idx1)
