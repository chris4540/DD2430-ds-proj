"""
Plot embedding space
"""
import os
import torch
import numpy as np
from utils.datasets import DeepFashionDataset
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from config.deep_fashion import DeepFashionConfig as cfg
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from network.resnet import ResidualEmbNetwork
from os.path import join
# utils
from utils import extract_embeddings
from utils.plot_deep_fashion import plot_embeddings
# Search tree
from tqdm import tqdm
from annoy import AnnoyIndex
# matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

exp_folder = "exp_results/exp_siamcos"
# exp_folder = "exp_results/exp_siamese_m2_100"

# Mdoels
emb_net = ResidualEmbNetwork()
emb_net.load_state_dict(torch.load(join(exp_folder, "_emb_net_20.pth")))

# Dataset
trans = Compose([
    Resize(cfg.sizes), ToTensor(),
    Normalize(cfg.mean, cfg.std)
])
# train_ds = DeepFashionDataset(cfg.root_dir, 'train', transform=trans)
train_ds = DeepFashionDataset(cfg.root_dir, 'val', transform=trans)
samples = np.random.choice(len(train_ds), 5000, replace=False)
train_ds = Subset(train_ds, samples)
# test_ds = DeepFashionDataset(cfg.root_dir, 'test', transform=trans)

# Extract embedding vectors
load_kwargs = {
    'batch_size': 128,
    'num_workers': os.cpu_count()
}

# test_embs, _ = extract_embeddings(emb_net, DataLoader(test_ds, **load_kwargs))
embs, labels = extract_embeddings(emb_net, DataLoader(train_ds, **load_kwargs))

# translate them to cpu + numpy
embs = embs.cpu().numpy()
labels = labels.cpu().numpy()
# -----------------------------------------------------------------------------
from cuml.manifold import TSNE
tsne = TSNE(n_iter=1000, metric="euclidean")
projected_emb = tsne.fit_transform(embs)
fig = plot_embeddings(projected_emb, labels)
fig.savefig('tmp.png', bbox_inches='tight')