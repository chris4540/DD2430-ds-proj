"""
Playgroud to write one loop

1. Wrtie one loop
2. write training in a free-style with ignite
3. Final code refactoring

Ref:
https://github.com/pytorch/ignite/blob/master/examples/fast_neural_style/neural_style.py

# check it for how to write loss without final pass
https://pytorch.org/ignite/quickstart.html#id1

https://fam-taro.hatenablog.com/entry/2018/12/25/021346
"""
import torch
import torch.backends.cudnn as cudnn
# ---------
# Model


# GPU
if torch.cuda.is_available():
    device = 'cuda'
    cudnn.benchmark = True
else:
    device = 'cpu'

# Model
from network.resnet import ResidualEmbNetwork
from network.resnet import ResidualNetwork
from network.siamese import SiameseNet
emb_net = ResidualEmbNetwork()
siamese_net = SiameseNet(emb_net)
clsf_net = ResidualNetwork(emb_net=emb_net, nb_classes=15)
assert id(emb_net) == id(siamese_net.emb_net)
assert id(emb_net) == id(clsf_net.emb_net)

# Dataset
from utils.datasets import DeepFashionDataset
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from config.deep_fashion import DeepFashionConfig as cfg
from torch.utils.data import DataLoader
from utils.datasets import Siamesize
trans = Compose([Resize(cfg.sizes), ToTensor(),
                 Normalize(cfg.mean, cfg.std), ])
# dataset
train_ds = DeepFashionDataset(
    cfg.root_dir, 'train', transform=trans)
siamese_train_ds = Siamesize(train_ds)
# loader
loader_kwargs = {
    'pin_memory': True,
    'batch_size': 100,
    'num_workers': 4,
}
s_train_loader = DataLoader(siamese_train_ds, **loader_kwargs)

# Optim
import torch.optim as optim
optimizer = optim.Adam(clsf_net.parameters(),
                       lr=1e-4)
# Loss functions
from trainer.loss import ContrastiveLoss
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.nn import CrossEntropyLoss


class ContrastiveLoss(_Loss):

    def __init__(self, margin, average=False):
        super().__init__()
        self.margin = margin
        self.average = average

    def forward(self, input_, target):
        # Unpack
        out1, out2 = input_
        dist = torch.norm((out1 - out2), p=2, dim=1)
        dist_sq = dist.pow(2)
        sim_img_loss = dist_sq
        dissim_img_loss = F.relu(self.margin**2 - dist_sq)
        y = target.float()
        losses = y * sim_img_loss + (1. - y) * dissim_img_loss
        if self.average:
            ret = losses.mean()
        else:
            ret = losses.sum()
        return ret


con_loss_fn = ContrastiveLoss(margin=1)
cs_loss_fn = CrossEntropyLoss()


clsf_net.to(device)

for _ in range(2):
    for inputs, targets in s_train_loader:

        img1, img2 = inputs
        c1, c2, target = targets

        img1 = img1.to(device)
        img2 = img2.to(device)
        c1 = c1.to(device)
        c2 = c2.to(device)
        target = target.to(device)

        # compute loss
        emb_vec1, emb_vec2 = siamese_net((img1, img2))
        contrastive_loss = con_loss_fn((emb_vec1, emb_vec2), target)

        # clsf_loss1 = cs_loss_fn(emb_vec1, emb_vec2), targets)
        y1 = clsf_net.foward_from_emb_vec(emb_vec1)
        y2 = clsf_net.foward_from_emb_vec(emb_vec2)
        clsf_loss1 = cs_loss_fn(y1, c1)
        clsf_loss2 = cs_loss_fn(y2, c2)

        # back-prop
        loss = contrastive_loss + clsf_loss1 + clsf_loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loss = loss.float()
        with torch.no_grad():
            print("Loss: ", loss.float())
            print("contrastive_loss: ", contrastive_loss.float())
            print("clsf_loss1: ", clsf_loss1.float())
            print("clsf_loss2: ", clsf_loss2.float())
