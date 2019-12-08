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
# take the input args
import sys


exp_folder = sys.argv[1]
print("Experiment result folder:", exp_folder)

# Mdoels
emb_net = ResidualEmbNetwork()
emb_net.load_state_dict(torch.load(join(exp_folder, "_emb_net_20.pth")))

# Dataset
trans = Compose([
    Resize(cfg.sizes), ToTensor(),
    Normalize(cfg.mean, cfg.std)
])
train_ds = DeepFashionDataset(cfg.root_dir, 'val', transform=trans)
rnd_state = np.random.RandomState(200)
samples = rnd_state.choice(len(train_ds), 5000, replace=False)
train_ds = Subset(train_ds, samples)

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
print("Plotting T-sne....")
from cuml.manifold import TSNE
tsne = TSNE(n_iter=1000, metric="euclidean")
projected_emb = tsne.fit_transform(embs)
fig = plot_embeddings(projected_emb, labels)
png_fname = join(exp_folder, 't-sne.png')
fig.savefig(png_fname, bbox_inches='tight')
pdf_fname = join(exp_folder, 't-sne.pdf')
fig.savefig(pdf_fname, bbox_inches='tight')
# -----------------------------------------------------------------------------
print("Plotting PCA....")
from cuml import PCA
pca_float = PCA(n_components=2)
cudf = pca_float.fit_transform(embs)
projected_emb = cudf.to_pandas().to_numpy()
fig = plot_embeddings(projected_emb, labels)
png_fname = join(exp_folder, 'pca.png')
fig.savefig(png_fname, bbox_inches='tight')
pdf_fname = join(exp_folder, 't-sne.pdf')
fig.savefig(pdf_fname, bbox_inches='tight')