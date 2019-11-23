#!/usr/bin/env python
"""
Script to calculate the mean and std
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
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

if __name__ == "__main__":
    deep_fashion_root_dir = "./deepfashion_data"
    trans = Compose([Resize((224, 224)), ToTensor()])

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
    # --------------------
    # tensor([0.7511, 0.7189, 0.7069])
    # tensor([0.0652, 0.0717, 0.0737])
    # tensor([0.2554, 0.2679, 0.2715])
    # --------------------
