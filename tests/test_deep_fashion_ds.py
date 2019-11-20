import numpy as np
from utils.datasets import DeepFashionDataset
from PIL import Image
deep_fashion_root_dir = "./deepfashion_data"


def test_ds_len():
    # train data loader
    train_loader = DeepFashionDataset(deep_fashion_root_dir, 'train')
    assert len(train_loader) > 0
    # Validation loader
    val_loader = DeepFashionDataset(deep_fashion_root_dir, 'val')
    assert len(val_loader) > 0

    test_loader = DeepFashionDataset(deep_fashion_root_dir, 'test')
    assert len(test_loader) > 0


def test_get_item_from_ds():
    train_loader = DeepFashionDataset(deep_fashion_root_dir, 'train')

    # Get one item
    im, target = train_loader[100]

    # check if an image
    assert isinstance(im, Image.Image)

    # Check class label
    assert target > 0
