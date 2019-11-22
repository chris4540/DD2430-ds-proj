import numpy as np
from utils.datasets import DeepFashionDataset
from PIL import Image
deep_fashion_root_dir = "./deepfashion_data"


def test_ds_len():
    # train data loader
    train_ds = DeepFashionDataset(deep_fashion_root_dir, 'train')
    assert len(train_ds) > 0
    # Validation loader
    val_ds = DeepFashionDataset(deep_fashion_root_dir, 'val')
    assert len(val_ds) > 0

    test_ds = DeepFashionDataset(deep_fashion_root_dir, 'test')
    assert len(test_ds) > 0


def test_get_item_from_ds():
    train_ds = DeepFashionDataset(deep_fashion_root_dir, 'train')

    # Get one item
    im, target = train_ds[100]

    # check img
    assert isinstance(im, np.ndarray)

    # Check class label
    assert target > 0
