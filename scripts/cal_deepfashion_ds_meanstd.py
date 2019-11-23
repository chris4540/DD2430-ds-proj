#!/usr/bin/env python
"""
Script to calculate the mean and std
Usage:
    ./scripts/cal_deepfashion_ds_meanstd.py
"""
import os.path
import sys
cur_path = os.path.realpath(__file__)
cur_dir = os.path.dirname(cur_path)
parent_dir = cur_dir[:cur_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)
print(parent_dir)
# --------------------------------------------
from utils.datasets import DeepFashionDataset
from utils.preprocessing import StandardScaler
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

if __name__ == "__main__":
    deep_fashion_root_dir = "./deepfashion_data"
    trans = Compose([
        Resize((224, 224)),
        ToTensor(),
        # Normalize([0.7464, 0.7155, 0.7043], [0.2606, 0.2716, 0.2744]),  # For check against
        ])

    train_ds = DeepFashionDataset(
        deep_fashion_root_dir, 'train', transform=trans)
    loader = DataLoader(train_ds, batch_size=200, num_workers=2)

    scalar = StandardScaler()
    for imgs, _ in tqdm(loader):
        scalar.partial_fit(imgs)

    print("--------------------")
    print(scalar._mean)
    print(scalar._var)
    print(scalar._std)
    print("--------------------")
