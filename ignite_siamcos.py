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

https://github.com/pytorch/ignite/blob/master/examples/notebooks/VAE.ipynb
"""
import numpy as np
import torch
import torch.backends.cudnn as cudnn
# ---------
from ignite.engine import _prepare_batch
from ignite.engine.engine import Engine

max_epochs = 10
pin_memory = True

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
from torch.utils.data import Subset
from torch.utils.data import TensorDataset
from utils.datasets import Siamesize
trans = Compose([Resize(cfg.sizes), ToTensor(),
                 Normalize(cfg.mean, cfg.std), ])
# dataset
train_ds = DeepFashionDataset(
    cfg.root_dir, 'train', transform=trans)
val_ds = DeepFashionDataset(
    cfg.root_dir, 'val', transform=trans)
siamese_train_ds = Siamesize(train_ds)
siamese_val_ds = Siamesize(val_ds)
if False:  # For overfitting test
    train_samples = np.random.choice(len(train_ds), 300, replace=False)
    val_samples = np.random.choice(len(val_ds), 100, replace=False)
    train_ds = Subset(train_ds, train_samples)
    val_ds = Subset(val_ds, val_samples)
    siamese_train_ds = Subset(siamese_train_ds, train_samples)
    siamese_val_ds = Subset(siamese_val_ds, val_samples)

# loader
import os
batch_size = 128
loader_kwargs = {
    'pin_memory': pin_memory,
    'batch_size': batch_size,
    'num_workers': os.cpu_count()
}
s_train_loader = DataLoader(siamese_train_ds, **loader_kwargs, shuffle=True)
# train_loader = DataLoader(val_ds, **loader_kwargs)
val_loader = DataLoader(val_ds, **loader_kwargs)

# Optim
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
params = [*siamese_net.parameters(), *clsf_net.parameters()]
optimizer = optim.Adam(params, lr=5e-4, weight_decay=1e-5)
scheduler = CosineAnnealingLR(
    optimizer, T_max=len(train_ds) * max_epochs / batch_size, eta_min=1e-6)
# Loss functions
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.nn import CrossEntropyLoss
from trainer.loss import ContrastiveLoss

import numpy as np
# margin = np.sqrt(1000)
margin = np.sqrt(0.2)
con_loss_fn = ContrastiveLoss(margin=margin, average=True)
cs_loss_fn = CrossEntropyLoss()
# scale_factor = 0.5 * margin**2  # consider per batch, negative pair ~ m**2 / 2
scale_factor = 1
# Acc
import torch
from ignite.metrics import Accuracy
# from ignite import metrics


class SiameseNetSimilarityAccuracy(Accuracy):
    """
    Calculate the similarity of a siamese network

    Example:
        eval = create_embedding_engine(..., )
    """

    def __init__(self, margin, l2_normalize=False):
        super().__init__()
        self.margin = margin
        self.l2_normalize = l2_normalize

    @torch.no_grad()
    def update(self, output):
        # calculate output
        out1, out2 = output["emb_vecs"]

        if self.l2_normalize:
            out1 = F.normalize(out1, p=2, dim=1)
            out2 = F.normalize(out2, p=2, dim=1)

        _, _, is_similar = output["targets"]

        # calculate L2 vector norm over the embedding dim
        dist = torch.norm((out1 - out2), p=2, dim=1)
        # The squared distance
        dist_sq = dist.pow(2)

        # dist_sq < margin**2 => similar => 1
        pred = (dist_sq < self.margin**2).int()
        super().update((pred, is_similar))


# class Accuracy(metrics.Accuracy):

#     @torch.no_grad()
#     def update(self, output):
#         # y_pred, y = output
#         y_pred = output["cls_pred"]
#         y_true = output["cls_true"]
#         super().update((y_pred, y_true))


clsf_net.to(device)
siamese_net.to(device)


def _update(engine, batch):
    siamese_net.train()
    clsf_net.train()
    optimizer.zero_grad()
    x, targets = _prepare_batch(batch, device=device, non_blocking=pin_memory)
    c1, c2, _ = targets

    emb_vec1, emb_vec2 = siamese_net(x)

    l2_emb_vec1 = F.normalize(emb_vec1, p=2, dim=1)
    l2_emb_vec2 = F.normalize(emb_vec2, p=2, dim=1)

    contras_loss = con_loss_fn((l2_emb_vec1, l2_emb_vec2), targets)
    y1 = clsf_net(emb_vec1)
    y2 = clsf_net(emb_vec2)
    clsf_loss1 = cs_loss_fn(y1, c1)
    clsf_loss2 = cs_loss_fn(y2, c2)

    loss = contras_loss + (clsf_loss1 + clsf_loss2) * scale_factor
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        cls_pred = torch.cat([y1, y2], dim=0)
        cls_true = torch.cat([c1, c2], dim=0)
        clsf_loss = clsf_loss1 + clsf_loss2

    ret = {
        "loss": loss.item(),
        "con_loss": contras_loss.item(),
        "clsf_loss": clsf_loss.item(),
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
        "sim_acc": SiameseNetSimilarityAccuracy(margin=margin, l2_normalize=True),
        "clsf_acc": Accuracy(
            output_transform=lambda x: (x['cls_pred'], x['cls_true']))
    }

    for name, metric in metrics.items():
        metric.attach(engine, name)

    from ignite.contrib.handlers import ProgressBar
    pbar = ProgressBar()
    pbar.attach(engine, output_transform=lambda x: {
        'con_loss': x['con_loss'],
        'clsf_loss': x['clsf_loss']
    })

    from ignite.engine import Events
    # @engine.on(Events.ITERATION_COMPLETED)
    # def log_training_loss(engine):
    #     print("Epoch[{}] Loss: {:.2f}".format(
    #         engine.state.epoch, engine.state.output["loss"]))

    @engine.on(Events.ITERATION_COMPLETED)
    def take_scheduler_step(engine):
        scheduler.step()
        # print(scheduler.get_lr())

    @engine.on(Events.EPOCH_COMPLETED)
    def log_training_acc(engine):
        metrics = engine.state.metrics
        sim_acc = metrics['sim_acc']
        clsf_acc = metrics['clsf_acc']
        print("Epoch[{}] sim_acc: {:.2f}; clsf_acc {:.2f}".format(
            engine.state.epoch, sim_acc, clsf_acc))

    from ignite.engine import create_supervised_evaluator
    from ignite.metrics import Loss
    from utils import extract_embeddings
    from trainer.metrics import SiameseNetSimilarityAccuracy as SimilarityAccuracy
    siamese_evaluator = create_supervised_evaluator(
        siamese_net, device=device, non_blocking=pin_memory, metrics={
            # no a good approach
            'accuracy': SimilarityAccuracy(margin, l2_normalize=True),
            'loss': Loss(con_loss_fn)
        })
    pbar = ProgressBar()
    pbar.attach(siamese_evaluator)
    clsf_evaluator = create_supervised_evaluator(
        clsf_net, device=device, non_blocking=pin_memory, metrics={
            'accuracy': Accuracy(),
            'loss': Loss(CrossEntropyLoss())
        })

    @engine.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        # loader_kwargs = {
        #     'pin_memory': True,
        #     'num_workers': 4,
        #     'batch_size': 100,
        # }
        siamese_val_loader = DataLoader(siamese_val_ds, **loader_kwargs)

        # ----------------------------------
        siamese_evaluator.run(siamese_val_loader)
        avg_acc = siamese_evaluator.state.metrics['accuracy']
        print("run_validation: SimilarityAccuracy accuracy: {}".format(
            avg_acc))
        val_embs, val_labels = extract_embeddings(emb_net, val_loader)

        val_emb_ds = TensorDataset(val_embs, val_labels)
        clsf_evaluator.run(DataLoader(val_emb_ds, **loader_kwargs))
        metrics = clsf_evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']
        print("run_validation: clsf accuracy: {}, loss: {}".format(
            avg_accuracy, avg_loss))
        return
        # if engine.state.epoch % 5 != 0:
        # return
        # ----------------------------------------------------------------------
        # train_loader = DataLoader(train_ds, **loader_kwargs)
        # train_embs, train_labels = extract_embeddings(emb_net, train_loader)
        # emb_dim = train_embs.shape[1]
        # ----------------------------------
        # from annoy import AnnoyIndex
        # from tqdm import tqdm

        # t = AnnoyIndex(emb_dim, metric='euclidean')
        # n_trees = 100
        # for i, emb_vec in enumerate(train_embs):
        #     t.add_item(i, emb_vec.cpu().numpy())
        # # build a forest of trees
        # tqdm.write("Building ANN forest...")
        # t.build(n_trees)
        # # ----------------------------------
        # top_k_corrects = dict()
        # # Meassure Prec@[5, 10, 20, 30]
        # k_vals = [10, 30, 50, 100, 500, 1000]
        # for i, emb_vec in enumerate(val_embs):
        #     correct_cls = val_labels[i]
        #     for k in k_vals:
        #         idx = t.get_nns_by_vector(emb_vec.cpu().numpy(), k)
        #         top_k_classes = train_labels[idx]
        #         correct = torch.sum(top_k_classes == correct_cls)
        #         accum_corr = top_k_corrects.get(k, 0)
        #         top_k_corrects[k] = accum_corr + correct.item()
        # # -------------------------------------------------
        # # calculate back the acc
        # top_k_acc = dict()
        # for k in k_vals:
        #     top_k_acc[k] = top_k_corrects[k] / k / val_embs.shape[0]

        # tqdm.write(
        #     "Top K Retrieval Results - Epoch: {}  Avg top-k accuracy:"
        #     .format(engine.state.epoch)
        # )

        # for k in k_vals:
        #     tqdm.write("  Prec@{} = {:.2f}, Corrects@{} = {}".format(
        #         k, top_k_acc[k], k, top_k_corrects[k]))

    engine.run(s_train_loader, max_epochs=max_epochs)
