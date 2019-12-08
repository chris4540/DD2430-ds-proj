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

# Extract embedding vectors
load_kwargs = {
    'batch_size': 128,
    'num_workers': os.cpu_count()
}

test_embs, _ = extract_embeddings(emb_net, DataLoader(test_ds, **load_kwargs))
val_embs, _ = extract_embeddings(
    emb_net, DataLoader(val_ds, **load_kwargs))

# search tree building
search_tree = AnnoyIndex(emb_net.emb_dim, metric='euclidean')
for i, vec in enumerate(val_embs):
    search_tree.add_item(i, vec.cpu().numpy())
search_tree.build(100)


def plot_search_results(query_img_dix):
    n_search = 5
    # prepate the plot

    result_idx = search_tree.get_nns_by_vector(
        test_embs[query_img_dix].cpu().numpy(), n_search)
    fig, axs = plt.subplots(ncols=n_search + 1, figsize=(40, 40))
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    # plot the query img
    img, cat = test_ds.get_img_cat(query_img_dix)
    axs[0].imshow(img)
    axs[0].set_title(cat)
    for j, i in enumerate(result_idx):
        img, cat = val_ds.get_img_cat(i)
        axs[j + 1].imshow(img)
        axs[j + 1].set_title(cat)
    return fig


# ran_state = np.random.RandomState(100)
# sel_idx = ran_state.choice(len(test_ds), 5, replace=False)
# sel_idx = [4002, 8716, 9388, 3013, 7513]
sel_idx = [3012, 4002, 8716, 9388, 3013, 7513]
for i in sel_idx:
    fig = plot_search_results(i)
    png = join(exp_folder, 'search_idx_{}.png'.format(i))
    fig.savefig(png, bbox_inches='tight')
    pdf = join(exp_folder, 'search_idx_{}.pdf'.format(i))
    fig.savefig(pdf, bbox_inches='tight')
