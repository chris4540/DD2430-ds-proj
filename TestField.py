#!/usr/bin/env python
from trainer.fashion_mnist import SiameseFashionMNISTTrainer
from config.fashion_mnist import FashionMNISTConfig
from utils.datasets import FashionMNIST
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from utils import extract_embeddings
from annoy import AnnoyIndex



trainer = SiameseFashionMNISTTrainer(
    log_interval=5, lr=1e-2, epochs=1, batch_size=100)
trainer.run()

cfg = FashionMNISTConfig

# data transform
data_transform = Compose([
        ToTensor(),
        Normalize((cfg.mean,), (cfg.std,))
    ])
ds_kwargs = {
    'root': cfg.root,
    'transform': data_transform
}

train_ds = FashionMNIST(train=True, download=True, **ds_kwargs)
train_loader = DataLoader(train_ds, shuffle=False, batch_size=100)

val_ds = FashionMNIST(train=False, download=False, **ds_kwargs)
val_loader = DataLoader(val_ds, shuffle=False, batch_size=100)


train_embs, train_labels = extract_embeddings(trainer.model, train_loader)

t = AnnoyIndex(50, metric='euclidean')
n_trees = 100
for i, emb_vec in enumerate(train_embs):
    t.add_item(i, emb_vec)
# build a forest of trees
t.build(n_trees)


# Test
val_embs, val_labels = extract_embeddings(trainer.model, train_loader)
correct = 0
cnt = 0
for i, emb_vec in enumerate(val_embs):
    correct_cls = val_labels[i]
    idx = t.get_nns_by_vector(val_embs[0], 5)
    top_k_classes = train_labels[idx]
    correct += np.sum(top_k_classes == correct_cls)
    cnt += len(idx)

    if cnt > 10000:
        break

print(correct / cnt)
