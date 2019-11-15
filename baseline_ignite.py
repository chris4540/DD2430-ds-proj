"""
Implementation using pytorch ignite

Ref:
    https://github.com/pytorch/ignite/blob/v0.2.1/examples/mnist/mnist.py
"""
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torchvision.datasets import FashionMNIST


class HyperParams:
    batch_size = 100


class FashionMNISTConfig:
    root = "./"
    mean = 0.28604059698879553
    std = 0.35302424451492237


class BaselineFashionMNIST:
    """
    Test tube class for constructing embbeding space only with classifcation
    method
    """

    @staticmethod
    def get_data_loaders():
        """
        Our target is to construct embbeding space. Therefore we use the "test set"
        as validation
        """
        # alias
        cfg = FashionMNISTConfig

        # data transform
        data_transform = Compose(
            [ToTensor(), Normalize((cfg.mean,), (cfg.std,))])

        # ----------------------------
        # Consturct data loader
        # ----------------------------
        ds_kwargs = {
            'root': cfg.root,
            'transform': data_transform
        }
        train_ds = FashionMNIST(train=True, download=True, **ds_kwargs)
        val_ds = FashionMNIST(train=False, download=False, **ds_kwargs)
        # ----------------------------
        # Consturct loader
        # ----------------------------
        train_loader = DataLoader(
            train_ds, shuffle=True, batch_size=HyperParams.batch_size)
        val_loader = DataLoader(val_ds, shuffle=False,
                                batch_size=HyperParams.batch_size)
        return train_loader, val_loader

    def run(epochs, log_interval):
        pass
        
