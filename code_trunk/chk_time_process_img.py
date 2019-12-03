"""
Check the time for processing images only
"""
# Dataset
from utils.datasets import DeepFashionDataset
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from config.deep_fashion import DeepFashionConfig as cfg
from torch.utils.data import DataLoader
from utils.datasets import Siamesize
from time import time
trans = Compose([Resize(cfg.sizes), ToTensor(),
                 Normalize(cfg.mean, cfg.std), ])
# dataset
train_ds = DeepFashionDataset(
    cfg.root_dir, 'train', transform=trans)

siamese_train_ds = Siamesize(train_ds)
loader_kwargs = {
    'pin_memory': True,
    'batch_size': 100,
    'num_workers': 16,
}
s_train_loader = DataLoader(siamese_train_ds, **loader_kwargs)

device = "cuda"
for _ in range(1):
    s_time = time()
    for inputs, targets in s_train_loader:
        img1, img2 = inputs
        c1, c2, target = targets

        # img1 = img1.to(device)
        # img2 = img2.to(device)
        # c1 = c1.to(device)
        # c2 = c2.to(device)
        # target = target.to(device)
        print("Time Escape = ", time() - s_time)
    e_time = time()
    print("Time per epoch for img processing = ", e_time - s_time)
