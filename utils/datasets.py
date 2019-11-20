"""
Reference:
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""
import numpy as np
import pandas as pd
from os.path import join as path_join
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import zipfile


class DeepFashionDataset(Dataset):
    """
    Abstracted data represetation for deep fashion dataset

    Responsible:
        1. manage meta data
        2. apply valid transformation

    Example:
        train_ds = DeepFashionDataset("./deepfashion_data")
        loader = DataLoader(train_ds)
    """

    metadata_csv = "deepfashion1_categoryData.csv"

    def __init__(self, root, ds_type, transform=None):
        """
        Args:
            root (str): The root directory of data
            ds_type (str): dataset type, either train, val, or test
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # check input if valid
        valid_ds_types = ['train', 'val', 'test']
        if ds_type not in valid_ds_types:
            raise ValueError(
                "ds_type should be one of the following: \"{}\"".format(
                    ', '.join(valid_ds_types)))

        # record down the init. args
        self.ds_type = ds_type
        self.transform = transform

        if self.ds_type == 'train':
            self.train = True
        else:
            self.train = False
        # ------------------------------------
        # Read the csv
        metadata_csv_file = path_join(root, self.metadata_csv)
        self._alldata = pd.read_csv(metadata_csv_file)

        # Select images, label from _alldata where dataset == ds_type
        self.data = self._alldata[
            self._alldata['dataset'] == self.ds_type][['images', 'label']]

        # Load the zip file
        self.img_zip = zipfile.ZipFile(path_join(root, "img.zip"), 'r')

    def __del__(self):
        """
        Destructor of this class
        """
        # Release the resources when this class destroy
        try:
            self.img_zip.close()
        except:
            pass

    def __len__(self):
        """
        Return the size of the dataset
        """
        ret = self.data.shape[0]
        return ret

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        metadata = self.data.iloc[idx]
        img_path = metadata['images']
        target = metadata['label']

        with self.img_zip.open(img_path) as f:
            # Open the image file
            img = Image.open(f)
            # load it as the Image class do lazy evaluation
            img.load()
            # Transform if applicable
            if self.transform:
                img = self.transform(img)

        return (img, target)
# -------------------------------------------------------------------------------


class Pairize(Dataset):
    """
    To make a dataset return pairs from another dataset

    Notes:
        This class is working in progress

    Example:
        ds = DeepFashionDataset(...)
        paired_ds = Pairize(ds)
        (img1, img2), (t1, t2) = paired_ds[0]
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
        self._pair_idx_set = set()

        # ----------------
        # Random state
        # ----------------
        if self.train:
            # just alias the numpy random module
            self.random_state = np.random
        else:
            # Use our random state container to provide sampling function
            self.random_state = np.random.RandomState(self._eval_seed)

    def _save_pair(self, idx_pair):
        pass

    def _if_used(self, idx_pair):
        pass


class SiameseMNIST(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

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

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(
                                   self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                   np.random.choice(
                                       list(
                                           self.labels_set - set([self.test_labels[i].item()]))
                                   )
                               ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
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
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.mnist_dataset)


class TripletMNIST(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(
                             self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                             np.random.choice(
                                 list(
                                     self.labels_set - set([self.test_labels[i].item()]))
                             )
                         ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item(
            )
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(
                    self.label_to_indices[label1])
            negative_label = np.random.choice(
                list(self.labels_set - set([label1])))
            negative_index = np.random.choice(
                self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

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
