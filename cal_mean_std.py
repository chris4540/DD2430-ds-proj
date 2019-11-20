"""
Script to calculate the mean and std
"""
from utils.datasets import DeepFashionDataset
from utils.datasets import Siamesize
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm


class StandardScaler:
    """
    Minic:
    sklearn.preprocessing.StandardScaler
    """

    def __init__(self):
        self.n_samples_seen_ = 0
        self._partial_sum = None                # i.e. mean * N
        self._partial_sum_of_sq_of_dev = None   # i.e. variance * N
        self._mean = 0
        self._var = 0
        self._std = 0

    def partial_fit(self, data):
        """
        The algorithm for incremental mean and std is given in Equation 1.5a,b
        in Chan, Tony F., Gene H. Golub, and Randall J. LeVeque.
        “Algorithms for computing the sample variance: Analysis and recommendations.”
        The American Statistician 37.3 (1983): 242-247:
        """
        # Calcualte this batch statistics
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)

        batch_mean = data.mean(2).mean(0)
        # print(batch_mean)
        batch_sum = batch_mean * batch_samples

        batch_var = torch.var(data, dim=2).mean(0)
        # print(batch_var)
        batch_sum_of_sq_of_dev = batch_var * batch_samples

        # -------------------------------------
        # Update the partial results
        # Partial sum of deviation square (S term)
        if self._partial_sum_of_sq_of_dev is None:
            self._partial_sum_of_sq_of_dev = batch_sum_of_sq_of_dev
        else:
            # Alias
            m = self.n_samples_seen_
            n = batch_samples
            # the second term of R.H.S of 1.5b
            extra_term = (self._partial_sum * n / m - batch_sum)**2
            extra_term = extra_term * m / (n * (m + n))
            self._partial_sum_of_sq_of_dev += batch_sum_of_sq_of_dev + extra_term

        # Partial sum (T term)
        if self._partial_sum is None:
            self._partial_sum = batch_sum
        else:
            self._partial_sum += batch_sum

        # n_samples_seen_
        self.n_samples_seen_ += batch_samples
        # -------------------------------------
        # Update mean, std, var
        self._mean = self._partial_sum / self.n_samples_seen_
        self._var = self._partial_sum_of_sq_of_dev / self.n_samples_seen_
        self._std = self._var.sqrt()


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
