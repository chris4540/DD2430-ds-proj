"""
Reference:
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

TODO:
    Documentaion
"""
import numpy as np
import pandas as pd
from os.path import join as path_join
from PIL import Image
from torch.utils.data import Dataset
from . import PairIndexSet


class DeepFashionDataset(Dataset):
    """
    Abstracted data represetation for deep fashion dataset

    Responsible:
        1. manage meta data
        2. apply valid transformation

    Example:
        trans = Compose([Resize((224,224)), ToTensor()])
        train_ds = DeepFashionDataset("./deepfashion_data", transform=trans)
        loader = DataLoader(train_ds)
    """

    metadata_csv = "sampled_data_meta.csv"

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
        self.root = root

        if self.ds_type == 'train':
            self.train = True
        else:
            self.train = False
        # ------------------------------------
        # Read the csv
        metadata_csv_file = path_join(root, self.metadata_csv)
        self._alldata = pd.read_csv(metadata_csv_file)

        # Select images, label from _alldata where dataset == ds_type
        sel_cols = ['images', 'label', 'category']
        self.data = self._alldata[
            self._alldata['dataset'] == self.ds_type][sel_cols]

        # reset the index to enable access with index
        self.data.reset_index(drop=True, inplace=True)

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
        metadata = self.data.loc[idx]
        img_path = metadata['images']
        target = metadata['label']
        img_full_path = path_join(self.root, img_path)
        with open(img_full_path, 'rb') as f:
            with Image.open(f) as img_file:
                if self.transform:
                    img = self.transform(img_file)
                else:
                    img = np.asarray(img_file)
        return (img, target)

    def get_img_cat(self, idx):
        """
        Get PIL image and category
        """
        metadata = self.data.loc[idx]
        img_path = metadata['images']
        category = metadata['category']
        img_full_path = path_join(self.root, img_path)
        with open(img_full_path, 'rb') as f:
            with Image.open(f) as img_file:
                img = np.asarray(img_file)
        ret = (img, category)
        return ret

    @classmethod
    def set_meta_csv(cls, csvfile):
        cls.metadata_csv = csvfile

    @property
    def classes(self):
        """
        Return a list of classes that the data has.
        """
        if not hasattr(self, '_unique_classes'):
            # build when we don't have
            self._unique_classes = self.data['label'].unique()
            self._unique_classes.sort()

        ret = self._unique_classes
        return ret

    def get_label_to_idxs(self, label):
        if not hasattr(self, '_label_to_idxs'):
            inv_map = {}
            for k, v in self.data['label'].items():
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
        metadata = self.data.loc[idx]
        target = metadata['label']
        return target


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

    # max number of recusive drawing
    n_trial = 5

    def __init__(self, dataset):
        """
        Args:
            dataset (DataSet): The dataset to be pairized
        """

        self._dataset = dataset

        # ------------------------------------------------
        # pair indices set; for recording returned pairs
        # ------------------------------------------------
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
            # build pair idx and is similar onces
            self._build_pairs_for_eval()

    def _build_pairs_for_eval(self):
        """
        Prebuild a list of tuple for evalation
        """
        rec = list()
        for idx1 in range(len(self)):
            idx2, is_similar = self._get_sec_idx_and_is_similar(idx1)
            rec.append((idx2, is_similar))
        self._pairs_for_eval = rec

    @property
    def train(self):
        return self._dataset.train

    def __getitem__(self, idx1):

        # get the second item idx of the pair and the target val
        idx2, target = self._get_idx2_and_target(idx1)

        # return the img and class
        img1, c1 = self._dataset[idx1]
        img2, c2 = self._dataset[idx2]
        return (img1, img2), (c1, c2, target)

    def _get_idx2_and_target(self, idx1):
        if self.train:
            idx2, target = self._get_sec_idx_and_is_similar(idx1)
        else:
            idx2, target = self._pairs_for_eval[idx1]

        return idx2, target

    def _get_sec_idx_and_is_similar(self, idx1, recur_cnt=0):
        """
        Args:
            idx (int):
            recur_cnt (int): recursion count
        """
        # -----------------------------------
        # Handel limit case of recursion
        # -----------------------------------
        if recur_cnt > self.n_trial:
            print("cleaning...")
            # clean the pair indices set
            self._pair_idx_set.clear()
            # Sample it again from cnt = 0
            return self._get_sec_idx_and_is_similar(idx1, 0)

        # generate target
        is_similar = self.random_state.choice([0, 1])

        # Get the second item for pairing
        idx2 = self._get_another_idx(idx1, is_similar)

        # Check if pairs are identical or already sampled
        if (idx1 == idx2) or (idx1, idx2) in self._pair_idx_set:
            # fail case, draw again by recursion
            return self._get_sec_idx_and_is_similar(idx1, recur_cnt + 1)
        else:
            # Successful case, add the pair to record set and return when training
            if self.train:
                self._pair_idx_set.add((idx1, idx2))

            return idx2, is_similar

    def _get_another_idx(self, idx1, is_similar):
        # Get the class of idx1 first
        c1 = self._dataset.get_idx_to_target(idx1)

        if is_similar:
            # similar if the pair are in the same class
            idxs = self._dataset.get_label_to_idxs(c1)

            # Therefore, the class of self._dataset[idxs] are c1
            idx2 = self.random_state.choice(idxs)
        else:
            # select the second element classs first, where c1 must not be c2
            classes = self._dataset.classes
            classes = list(classes)  # make a copy
            classes.remove(c1)

            c2 = self.random_state.choice(classes)
            idxs = self._dataset.get_label_to_idxs(c2)
            idx2 = self.random_state.choice(idxs)

        return idx2

    def __len__(self):
        return len(self._dataset)
