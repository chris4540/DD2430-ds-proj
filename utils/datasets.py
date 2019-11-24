# alias
from ._deep_fashion_ds import DeepFashionDataset
from ._deep_fashion_ds import Siamesize
from ._fashion_mnist import SiameseMNIST
from ._fashion_mnist import BalancedBatchSampler
import torchvision
import torch

class FashionMNIST(torchvision.datasets.FashionMNIST):
    """
    Add functions for enabling to be Siamesize
    """

    @property
    def classes(self):
        """
        Return a list of classes that the data has.
        """
        if not hasattr(self, '_unique_classes'):
            # build when we don't have
            self._unique_classes = torch.unique(self.targets).numpy().tolist()

        ret = self._unique_classes

        return ret

    def get_label_to_idxs(self, label):
        if not hasattr(self, '_label_to_idxs'):
            inv_map = {}
            for k,v in enumerate(self.targets.numpy()):
                inv_map[v] = inv_map.get(v, [])
                inv_map[v].append(k)

            self._label_to_idxs = inv_map

        ret = self._label_to_idxs[label]
        return ret

    def get_idx_to_target(self, idx):
        """
        Get the class (target) of the data by selected index

        Args:
            idx (int): the index of the data

        Return:
            the class (probabaly the class index)

        Notes:
            This function is to skip the transform of img calculation
        """
        ret = self.targets[idx]
        ret = int(ret)
        return ret
