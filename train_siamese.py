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
from utils.loss import ContrastiveLoss
from utils.datasets import SiameseMNIST
import torch.nn as nn


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def fwd_to_emb_layer(self, x):
        return self.embedding_net(x)


mean, std = 0.28604059698879553, 0.35302424451492237
batch_size = 128

# DataSet
train_dataset = FashionMNIST('./FashionMNIST', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((mean,), (std,))
                             ]))
test_dataset = FashionMNIST('./FashionMNIST', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((mean,), (std,))
                            ]))
# Returns pairs of images and target same/different
siamese_train_ds = SiameseMNIST(train_dataset)
siamese_test_ds = SiameseMNIST(test_dataset)

has_cuda = torch.cuda.is_available()
# kwargs = {'num_workers': 1, 'pin_memory': True} if has_cuda else {}
kwargs = {}

# data loders
siamese_train_loader = torch.utils.data.DataLoader(
    siamese_train_ds, batch_size=batch_size, shuffle=True, **kwargs)
siamese_test_loader = torch.utils.data.DataLoader(
    siamese_test_ds, batch_size=batch_size, shuffle=False, **kwargs)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
# ---------------------------------------------------------------------

# Step 2
embedding_net = SimpleCNN(last_layer='emb')
# Step 3
model = SiameseNet(embedding_net)

margin = 1.
loss_fn = ContrastiveLoss(margin)

lr = 1e-2
if has_cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 50

fit(siamese_train_loader, siamese_test_loader, model, loss_fn, optimizer, scheduler,
    n_epochs, has_cuda, log_interval)
# ---------------------------------------------------------------------------
# Obtain the embeddings
embeddings, labels = extract_embeddings(embedding_net, test_loader)

tsne = TSNE(random_state=1, n_iter=1000, metric="cosine")

projected_emb = tsne.fit_transform(embeddings)

fig = plot_embeddings(projected_emb, labels)

fig.savefig('siamese.png', bbox_inches='tight')
