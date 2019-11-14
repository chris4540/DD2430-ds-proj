from sklearn.manifold import TSNE
import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from utils.metrics import AccumulatedAccuracyMetric
from utils.trainer import fit
from network.simple_cnn import SimpleCNN
from torch.optim import lr_scheduler
import torch.optim as optim
from utils import extract_embeddings
from sklearn.decomposition import PCA
from utils.plot_fashion_minst import plot_embeddings

mean, std = 0.28604059698879553, 0.35302424451492237
batch_size = 256

# DataSet
train_dataset = FashionMNIST('../data/FashionMNIST', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((mean,), (std,))
                             ]))
test_dataset = FashionMNIST('../data/FashionMNIST', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((mean,), (std,))
                            ]))

has_cuda = torch.cuda.is_available()
# kwargs = {'num_workers': 1, 'pin_memory': True} if has_cuda else {}
kwargs = {}

# data loders
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

n_classes = 10
# ---------------------------------------------------------------------

model = SimpleCNN()
loss_fn = torch.nn.CrossEntropyLoss()
lr = 1e-2
if has_cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 1
log_interval = 50

fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler,
    n_epochs, has_cuda, log_interval, metrics=[AccumulatedAccuracyMetric()])
# ---------------------------------------------------------------------------
# Obtain the embeddings
embeddings, labels = extract_embeddings(model, test_loader)
# # Project it with pca
# pca = PCA(n_components=2)
# projected_emb = pca.fit_transform(embeddings)

# The default of 1,000 iterations gives fine results, but I'm training for longer just to eke
# out some marginal improvements. NB: This takes almost an hour!
tsne = TSNE(random_state=1, n_iter=1000, metric="cosine")

projected_emb = tsne.fit_transform(embeddings)

fig = plot_embeddings(projected_emb, labels)

fig.savefig('baseline.png', bbox_inches='tight')
