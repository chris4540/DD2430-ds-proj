import torch
import numpy as np
from utils.datasets import DeepFashionDataset
from utils.datasets import Siamesize
from config.deep_fashion import root_dir

def test_siamesize_train_ds_basic():
	dataset = DeepFashionDataset(root_dir, 'train')
	siamese_ds = Siamesize(dataset)

	targets = list()
	for idx1 in range(10000):
		idx2, is_similar = siamese_ds._get_paired_idx_and_is_similar(idx1)
		assert is_similar in [0, 1]
		assert idx1 != idx2
		# store the targets
		targets.append(is_similar)

	# empirically check probability of the target getting 0 or 1
	prob = np.mean(targets)
	assert 0.4 <= prob <= 0.6
