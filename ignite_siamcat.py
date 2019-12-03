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
from ignite.engine import _prepare_batch
from ignite.engine.engine import Engine


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
from network.clsf_net import ClassificationNet

emb_net = ResidualEmbNetwork()
siamese_net = SiameseNet(emb_net)
clsf_net = ClassificationNet(emb_net.emb_dim, nb_classes=15)

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
val_ds = DeepFashionDataset(
    cfg.root_dir, 'val', transform=trans)
siamese_train_ds = Siamesize(train_ds)
# loader
loader_kwargs = {
    'pin_memory': True,
    'batch_size': 100,
    'num_workers': 4,
}
s_train_loader = DataLoader(siamese_train_ds, **loader_kwargs)
train_loader = DataLoader(val_ds, **loader_kwargs)
val_loader = DataLoader(val_ds, **loader_kwargs)

# Optim
import torch.optim as optim
params = [*siamese_net.parameters(), *clsf_net.parameters()]
optimizer = optim.Adam(params, lr=1e-3)
# Loss functions
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.nn import CrossEntropyLoss
from trainer.loss import ContrastiveLoss


con_loss_fn = ContrastiveLoss(margin=1)
cs_loss_fn = CrossEntropyLoss()

# Acc
import torch
# from ignite.metrics import Accuracy
from ignite import metrics


class SiameseNetSimilarityAccuracy(metrics.Accuracy):
    """
    Calculate the similarity of a siamese network

    Example:
        eval = create_embedding_engine(..., )
    """

    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    @torch.no_grad()
    def update(self, output):
        # calculate output
        out1, out2 = output["emb_vecs"]

        _, _, is_similar = output["targets"]

        # calculate L2 vector norm over the embedding dim
        dist = torch.norm((out1 - out2), p=2, dim=1)
        # The squared distance
        dist_sq = dist.pow(2)

        # dist_sq < margin**2 => similar => 1
        pred = (dist_sq < self.margin**2).int()
        super().update((pred, is_similar))


class Accuracy(metrics.Accuracy):

    @torch.no_grad()
    def update(self, output):
        # y_pred, y = output
        y_pred = output["cls_pred"]
        y_true = output["cls_true"]
        super().update((y_pred, y_true))


clsf_net.to(device)
siamese_net.to(device)


def _update(engine, batch):
    siamese_net.train()
    clsf_net.train()
    optimizer.zero_grad()
    x, targets = _prepare_batch(batch, device=device)
    c1, c2, _ = targets

    emb_vec1, emb_vec2 = siamese_net(x)
    contras_loss = con_loss_fn((emb_vec1, emb_vec2), targets)
    y1 = clsf_net(emb_vec1)
    y2 = clsf_net(emb_vec2)
    clsf_loss1 = cs_loss_fn(y1, c1)
    clsf_loss2 = cs_loss_fn(y2, c2)

    loss = contras_loss + clsf_loss1 + clsf_loss2
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        cls_pred = torch.cat([y1, y2], dim=0)
        cls_true = torch.cat([c1, c2], dim=0)

    ret = {
        "loss": loss.item(),
        "contrastive_loss": contras_loss.item(),
        "emb_vecs": [emb_vec1, emb_vec2],
        "cls_pred": cls_pred,
        "cls_true": cls_true,
        "targets": targets
    }
    return ret


# ----------------------------------------
if __name__ == "__main__":
    engine = Engine(_update)
    metrics = {
        "sim_acc": SiameseNetSimilarityAccuracy(margin=1),
        "clsf_acc": Accuracy()
    }

    for name, metric in metrics.items():
        metric.attach(engine, name)

    from ignite.contrib.handlers import ProgressBar
    pbar = ProgressBar()
    pbar.attach(engine, output_transform=lambda x: {'loss': x['loss']})

    from ignite.engine import Events
    # @engine.on(Events.ITERATION_COMPLETED)
    # def log_training_loss(engine):
    #     print("Epoch[{}] Loss: {:.2f}".format(
    #         engine.state.epoch, engine.state.output["loss"]))

    @engine.on(Events.EPOCH_COMPLETED)
    def log_training_acc(engine):
        metrics = engine.state.metrics
        sim_acc = metrics['sim_acc']
        clsf_acc = metrics['clsf_acc']
        print("Epoch[{}] sim_acc: {:.2f}; clsf_acc {:.2f}".format(
            engine.state.epoch, sim_acc, clsf_acc))

    engine.run(s_train_loader, max_epochs=1)
