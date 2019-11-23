import numpy as np
from utils.datasets import DeepFashionDataset
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torch.utils.data import Subset

deep_fashion_root_dir = "./deepfashion_data"


##################
# Helper function
##################
def assert_dataset(dataset_, n_test=3):
    for i in np.random.choice(len(dataset_), n_test):
        # Get one item
        im, target = dataset_[i]

        # check img
        assert isinstance(im, np.ndarray)

        # Check class label
        assert target >= 0


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
    assert_dataset(train_ds)

def test_subset_from_ds():
    train_ds = DeepFashionDataset(deep_fashion_root_dir, 'train')
    trans = Compose([Resize((224, 224)), ToTensor()])
    n_samples = 20
    select_idx = np.random.choice(len(train_ds), n_samples, replace=False)
    sub_ds = Subset(train_ds, select_idx)
    assert len(sub_ds) == n_samples
    assert_dataset(sub_ds)


