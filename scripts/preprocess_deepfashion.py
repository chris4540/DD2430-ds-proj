#!/usr/bin/env python
from inspect import getsourcefile
import os.path
import sys
cur_path = os.path.abspath(getsourcefile(lambda:0))
cur_dir = os.path.dirname(cur_path)
parent_dir = cur_dir[:cur_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)
# ------------------------------------------
import torch
from tqdm import tqdm
from os.path import join as path_join
from utils.datasets import DeepFashionDataset
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torch.utils.data import DataLoader


# The simplest transformer
class Config:
	root_dir = "./deepfashion_data"
	trans = Compose([Resize((224, 224)), ToTensor()])

def get_img_path_to_tensor_dict(ds):
	ret = dict()
	desc = "Processing {} dataset".format(ds.ds_type)
	for i in tqdm(range(len(ds)), desc=desc):
		img, _ = ds[i]
		metadata = ds.get_metadata(i)
		img_path = metadata['images']
		ret[img_path] = img
	return ret

def preprocess(ds_type):
	ds = DeepFashionDataset(Config.root_dir, ds_type, transform=Config.trans)
	img_dict = get_img_path_to_tensor_dict(ds)
	torch.save(test_img_dict, path_join(Config.root_dir, 'img_{}.pkl'.format(ds_type)))


if __name__ == '__main__':
	preprocess('train')
	preprocess('val')
	preprocess('test')
