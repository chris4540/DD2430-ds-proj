import torch
import numpy as np
from utils.datasets import DeepFashionDataset
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torch.utils.data import DataLoader
from torch.utils.data import Subset

deep_fashion_root_dir = "./deepfashion_data"
trans = Compose([Resize((224, 224)), ToTensor()])
train_ds = DeepFashionDataset(
            deep_fashion_root_dir, 'train', transform=trans)

# subset
n_samples = 200
sel_idx = np.random.choice(list(range(len(train_ds))), n_samples, replace=False)
assert len(set(sel_idx)) == n_samples
ds = Subset(train_ds, sel_idx)

