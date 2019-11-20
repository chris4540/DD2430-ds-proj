from utils.datasets import DeepFashionDataset
from utils.datasets import Siamesize
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torch.utils.data import DataLoader
import torch

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    deep_fashion_root_dir = "./deepfashion_data"
    trans = Compose([Resize((224, 224)), ToTensor()])
    torch.multiprocessing.freeze_support()

    train_ds = DeepFashionDataset(
        deep_fashion_root_dir, 'train', transform=trans)
    siam_ds = Siamesize(train_ds)
    loader = DataLoader(siam_ds, batch_size=200)

    for _ in range(20):
        for (img1, img2), (c1, c2, target) in loader:
            print(c1.shape)
