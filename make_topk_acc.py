"""
i think maybe vivek tell you something wrong. please do this:

you can sample 100-300 from each of the 15 classes to calculate the mean and std dev of top-50 within each class.

the sampling will be faster than entire dataset and still reliable.
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
# Search tree
from tqdm import tqdm
from annoy import AnnoyIndex
# matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
# take the input args
import sys
# ---------------------------------------------
cat_to_idx = {
    "Blazer":        0,
    "Blouse":        1,
    "Cardigan":      2,
    "Dress":         3,
    "Jacket":        4,
    "Jeans":         5,
    "Jumpsuit":      6,
    "Leggings":      7,
    "Romper":        8,
    "Shorts":        9,
    "Skirt":        10,
    "Sweater":      11,
    "Tank":         12,
    "Tee":          13,
    "Top":          14,
}
idx_to_cat = {v: k for k, v in cat_to_idx.items()}

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
# train_ds = DeepFashionDataset(cfg.root_dir, 'train', transform=trans)
val_ds = DeepFashionDataset(cfg.root_dir, 'val', transform=trans)
test_ds = DeepFashionDataset(cfg.root_dir, 'test', transform=trans)
n_samples_per_class = 100
rnd_state = np.random.RandomState(200)
samples = rnd_state.choice(len(test_ds), n_samples_per_class*len(test_ds.classes), replace=False)
test_ds = Subset(test_ds, samples)

# Extract embedding vectors
load_kwargs = {
    'batch_size': 128,
    'num_workers': os.cpu_count()
}

test_embs, test_labels = extract_embeddings(emb_net, DataLoader(test_ds, **load_kwargs))
val_embs, val_labels = extract_embeddings(
    emb_net, DataLoader(val_ds, **load_kwargs))

# search tree building
search_tree = AnnoyIndex(emb_net.emb_dim, metric='euclidean')
for i, vec in enumerate(val_embs):
    search_tree.add_item(i, vec.cpu().numpy())
search_tree.build(100)

# top_k_corrects = dict()

results = list()

n_search = 50
for cls in val_ds.classes:
    indices = np.where(test_labels == cls)[0]
    top_k_accs = list()
    for i in indices:
        emb_vec = test_embs[i]
        correct_cls = test_labels[i]
        idx = search_tree.get_nns_by_vector(emb_vec, n_search)
        top_k_classes = val_labels[idx]
        correct = torch.sum(top_k_classes == correct_cls).item()
        top_k_accs.append(correct / n_search)
    mean = np.mean(top_k_accs)
    std = np.std(top_k_accs)
    ret = {
        "catagory": idx_to_cat[cls],
        "label": int(cls),
        "n_samples": len(indices),
        "k": int(n_search),
        "mean": np.round(mean, decimals=4),
        "std": np.round(std, decimals=4)
    }
    results.append(ret)

import json
file = join(exp_folder, 'search_acc_per_class.json')
with open(file, 'w') as f:
    json.dump(results, f, indent=2)
 