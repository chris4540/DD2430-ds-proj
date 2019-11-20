from utils.datasets import DeepFashionDataset


def test_ds_len():
    # train data loader
    train_loader = DeepFashionDataset("./deepfashion_data", 'train')
    assert len(train_loader) > 0
    # Validation loader
    val_loader = DeepFashionDataset("./deepfashion_data", 'val')
    assert len(val_loader) > 0

    test_loader = DeepFashionDataset("./deepfashion_data", 'test')
    assert len(test_loader) > 0


def test_get_item_from_ds():
    pass
