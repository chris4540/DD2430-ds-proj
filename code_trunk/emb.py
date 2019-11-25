import torch
from network.siamese import SiameseNet
from network.resnet import ResidualEmbNetwork
import os
import numpy as np
from utils.datasets import DeepFashionDataset
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from utils import extract_embeddings
import pickle
from cuml.manifold import TSNE


emb_net = ResidualEmbNetwork()
model = SiameseNet(emb_net)

trans = Compose(
    [
        Resize((224, 224)),
        ToTensor(),
        Normalize([0.7511, 0.7189, 0.7069], [0.2554, 0.2679, 0.2715]),

    ])

model.load_state_dict(torch.load('siamese_resnet18.pth'))
deep_fashion_root_dir = "./deepfashion_data"
train_ds = DeepFashionDataset(
    deep_fashion_root_dir, 'train', transform=trans)
emb_net = model.emb_net
emb_net.cuda()
# subset
n_samples = 25000
sel_idx = np.random.choice(
    list(range(len(train_ds))),
    n_samples, replace=False)

assert len(set(sel_idx)) == n_samples

ds = Subset(train_ds, sel_idx)
loader = DataLoader(
    ds, batch_size=100, pin_memory=True, num_workers=os.cpu_count())
print("extracting...")
embeddings, labels = extract_embeddings(emb_net, loader)

tsne = TSNE(n_iter=400, metric="euclidean")
projected_emb = tsne.fit_transform(embeddings)

with open('projected_emb.pkl', 'wb') as handle:
    pickle.dump(projected_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('labels.pkl', 'wb') as handle:
    pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
