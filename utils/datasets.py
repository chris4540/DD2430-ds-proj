"""
Reference:
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""
from . import PairIndexSet
import numpy as np
import pandas as pd
from os.path import join as path_join
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
# alias
from ._deep_fashion_ds import DeepFashionDataset


class Siamesize(Dataset):
    """
    To make a dataset return pairs from another dataset for Siamese training

    Notes:
        This class is working in progress

    Example:
        ds = DeepFashionDataset(...)
        siamese_ds = Siamesize(ds)
        loader = DataLoader(ds, batch_size=200, pin_memory=True)
        loader = DataLoader(ds, batch_size=200, pin_memory=True, num_workers=2)

    TODO:
        - Check edge case if two pairs are identical
    """

    # The random seed for selecting pairs in the mode of validation / testing
    # Since we should always to keep the validation and test are the same
    _eval_seed = 100

    def __init__(self, dataset):
        """
        Args:
            dataset (DataSet): The dataset to be pairized
        """
        self._dataset = dataset
        self.train = dataset.train

        # ----------------
        # pair index set
        # ----------------
        self._pair_idx_set = PairIndexSet()

        # ----------------
        # Random state
        # ----------------
        if self.train:
            # just alias the numpy random module
            self.random_state = np.random  # enable to set a gobal seed
        else:
            # Use our random state container to provide sampling function
            self.random_state = np.random.RandomState(self._eval_seed)
            self._build_eval_pairs()

    def _build_eval_pairs(self):
        """
        To build evaluation dataset which is balanced
        """
        pass

    def __getitem__(self, idx):
        """
        """
        if self.train:
            return self._get_item_in_train_mode(idx)

    def _get_item_in_train_mode(self, idx):
        """
        Args:
            idx (int):

        Outline:
            0.
            1. Select the pair with (idx, idx_2), where idx_2 is randomly selected
            2. Check if already sampled
            3. Goto 1 if sampled already, but limit the number of trials
            4. When the number of trial > sqrt(the size of the dataset), clean the PairIndexSet
            5. return (img1, img2), (c1, c2, target)
        """
        # randomly choose if the classes of the training pair are the same or not.
        target = self.random_state.choice([0, 1])

        # take the img1 and c1 with idx from self._dataset
        img1, c1 = self._dataset[idx]

        # draw the second data
        n_trial = 10
        for t in range(n_trial):
            if target == 1:
                idxs = self._dataset.get_label_to_idxs(c1)
                # Therefore, the class of self._dataset[idxs] are c1
                idx2 = self.random_state.choice(idxs)
            else:
                # if target == 0 then we select the class first
                classes = self._dataset.unique_classes
                classes = list(classes)  # make a copy
                classes.remove(c1)

                c2 = self.random_state.choice(classes)
                idxs = self._dataset.get_label_to_idxs(c2)
                idx2 = self.random_state.choice(idxs)

            if (idx, idx2) not in self._pair_idx_set:
                break
        # ---------------------------------------------------------------------
        if t < n_trial:
            img2, c2 = self._dataset[idx2]
            # Add the index pair
            self._pair_idx_set.add((idx, idx2))
            return (img1, img2), (c1, c2, target)
        else:
            print("Going to clean the cache and try resample it again")
            # clean the _pair_idx_set
            self._pair_idx_set.clear()
            return self._get_item_in_train_mode(idx)
    # --------------------------------------------------------------------------

    def __len__(self):
        return len(self._dataset)


class SiameseMNIST(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, mnist_dataset, rndState=13):
        self.mnist_dataset = mnist_dataset

        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        # to allow train and test with diff seeds
        self.rndState = rndState
        random_state = np.random.RandomState(rndState)

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            # increase randomness, reduce bias
            positive_pairs = [[i,
                               random_state.choice(
                                   self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in np.random.choice(len(self.test_data), len(self.test_data)//2, False)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                   np.random.choice(
                                       list(
                                           self.labels_set - set([self.test_labels[i].item()]))
                                   )
                               ]),
                               0]
                              for i in np.random.choice(len(self.test_data), len(self.test_data)//2, False)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item(
            )
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(
                        self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(
                    list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(
                    self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
            label2 = self.train_labels[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), (label1, label2, target)

    def __len__(self):
        return len(self.mnist_dataset)


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(
                self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                   class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
