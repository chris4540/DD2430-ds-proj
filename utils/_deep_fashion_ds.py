import numpy as np
import pandas as pd
from os.path import join as path_join
from PIL import Image
from torch.utils.data import Dataset

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

        # if self.ds_type == 'train' :
        self.train = True
        # else:
            # self.train = False
        # ------------------------------------
        # Read the csv
        metadata_csv_file = path_join(root, self.metadata_csv)
        self._alldata = pd.read_csv(metadata_csv_file)

        # Select images, label from _alldata where dataset == ds_type
        self.data = self._alldata[
            self._alldata['dataset'] == self.ds_type][['images', 'label']]

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

    @classmethod
    def set_meta_csv(cls, csvfile):
        cls.metadata_csv = csvfile

    @property
    def unique_classes(self):
        """
        Return a list of unique classes
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
