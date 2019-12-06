import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
# utils
from utils.hparams import HyperParams
from ignite.contrib.handlers import ProgressBar
# Networks
from network.resnet import ResidualEmbNetwork
from network.siamese import SiameseNet
from network.clsf_net import ClassificationNet
import torch.nn.functional as F
# datasets
from utils.datasets import DeepFashionDataset
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from config.deep_fashion import DeepFashionConfig as cfg
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from utils.datasets import Siamesize
# Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
# Loss
from torch.nn import CrossEntropyLoss
from utils.loss import ContrastiveLoss
# training
from ignite.engine import _prepare_batch
from ignite.engine.engine import Engine
from ignite.engine import Events
# metrics
from ignite.metrics import Accuracy
from utils.metrics import SiamSimAccuracy


class SiameseCosDistanceCat:
    """
    Example:
    >> exp = SiameseCosDistance()
    >> exp.run(max_epochs=10)
    """

    l2_normalize = True

    # opts to make small dataset for overfitting as debug purpose
    _debug = False

    #
    _models = None
    _datasets = None
    _optimizer = None
    _loss_fns = None

    def __init__(self, exp_folder=None, log_interval=1, **kwargs):
        self._hparams = HyperParams(**kwargs)
        self.hparams.display()

        # self.hparams.save_to_txt(self.exp_folder / 'hparams.txt')
        # self.hparams.save_to_json(self.exp_folder / 'hparams.json')

        self.batch_size = self.hparams.batch_size
        self.pin_memory = True
        self.loader_kwargs = {
            'pin_memory': self.pin_memory,
            'batch_size': self.batch_size,
            'num_workers': os.cpu_count()
        }
        if torch.cuda.is_available():
            self.device = 'cuda'
            cudnn.benchmark = True
        else:
            self.device = 'cpu'

        self.margin = 1

        self._debug = kwargs.get('debug', False)  # dict.get(k, default)

    # --------------------------------
    # Experiment definitions
    # --------------------------------

    @property
    def models(self):
        if self._models is None:
            emb_net = ResidualEmbNetwork()
            siamese_net = SiameseNet(emb_net)
            clsf_net = ClassificationNet(emb_net.emb_dim, nb_classes=15)
            self._models = {
                "emb_net": emb_net,
                "siam_net": siamese_net,
                "clsf_net": clsf_net,
            }

            for model in self.models.values():
                model.to(self.device)

        return self._models

    @property
    def datasets(self):
        if self._datasets is None:
            trans = Compose([
                Resize(cfg.sizes), ToTensor(),
                Normalize(cfg.mean, cfg.std)
            ])
            # dataset
            train_ds = DeepFashionDataset(
                cfg.root_dir, 'train', transform=trans)
            val_ds = DeepFashionDataset(
                cfg.root_dir, 'val', transform=trans)
            siam_train_ds = Siamesize(train_ds)
            siam_val_ds = Siamesize(val_ds)

            # Subset if needed
            if self._debug:
                train_samples = np.random.choice(
                    len(train_ds), 600, replace=False)
                val_samples = np.random.choice(len(val_ds), 200, replace=False)
                # Subset the datasets
                train_ds = Subset(train_ds, train_samples)
                val_ds = Subset(val_ds, val_samples)
                siam_train_ds = Subset(siam_train_ds, train_samples)
                siam_val_ds = Subset(siam_val_ds, val_samples)
            # -------------------------------------------------------
            # pack them up
            self._datasets = {
                "train": train_ds,
                "val": val_ds,
                "siam_train": siam_train_ds,
                "siam_val": siam_val_ds,
            }

        return self._datasets

    @property
    def optimizer(self):
        if self._optimizer is None:
            models = self.models
            siam_net = models['siam_net']
            clsf_net = models['clsf_net']
            params = [
                *siam_net.parameters(),
                *clsf_net.parameters()
            ]
            optimizer = optim.Adam(params, lr=5e-4, weight_decay=1e-5)
            self._optimizer = optimizer

        return self._optimizer

    @property
    def loss_fns(self):
        if not self._loss_fns:
            self._loss_fns = {
                'contrastive': ContrastiveLoss(margin=self.margin, average=True),
                'cross_entropy': CrossEntropyLoss()
            }
        return self._loss_fns

    @property
    def scale_factor(self):
        """
        For inherit, see the doc of
        SiameseEucDistanceCat.scale_factor for details
        """
        return 1.0

    @property
    def hparams(self):
        return self._hparams

    def train_update(self, engine, batch):
        # alias
        siam_net = self.models['siam_net']
        clsf_net = self.models['clsf_net']
        optimizer = self.optimizer
        con_loss_fn = self.loss_fns['contrastive']
        cs_loss_fn = self.loss_fns['cross_entropy']

        siam_net.train()
        clsf_net.train()
        optimizer.zero_grad()
        x, targets = _prepare_batch(batch, device=self.device,
                                    non_blocking=self.pin_memory)
        c1, c2, _ = targets

        emb_vec1, emb_vec2 = siam_net(x)

        if self.l2_normalize:
            l2_emb_vec1 = F.normalize(emb_vec1, p=2, dim=1)
            l2_emb_vec2 = F.normalize(emb_vec2, p=2, dim=1)
            contras_loss = con_loss_fn((l2_emb_vec1, l2_emb_vec2), targets)
        else:
            contras_loss = con_loss_fn((emb_vec1, emb_vec2), targets)

        y1 = clsf_net(emb_vec1)
        y2 = clsf_net(emb_vec2)
        clsf_loss1 = cs_loss_fn(y1, c1)
        clsf_loss2 = cs_loss_fn(y2, c2)

        loss = contras_loss + (clsf_loss1 + clsf_loss2) * self.scale_factor
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
            "cls_pred": cls_pred,
            "cls_true": cls_true,
            "targets": targets
        }

        # add the emb_vecs
        if self.l2_normalize:
            ret["emb_vecs"] = [l2_emb_vec1, l2_emb_vec2]
        else:
            ret["emb_vecs"] = [emb_vec1, emb_vec2]

        return ret

    def eval_inference(self, engine, batch):
        siam_net = self.models['siam_net']
        clsf_net = self.models['clsf_net']
        siam_net.eval()
        clsf_net.eval()
        with torch.no_grad():
            x, targets = _prepare_batch(batch, device=self.device,
                                        non_blocking=self.pin_memory)
            emb_vec1, emb_vec2 = siam_net(x)

            if self.l2_normalize:
                l2_emb_vec1 = F.normalize(emb_vec1, p=2, dim=1)
                l2_emb_vec2 = F.normalize(emb_vec2, p=2, dim=1)

            # make inference with emb_vecs
            # predictions
            y1 = clsf_net(emb_vec1)
            y2 = clsf_net(emb_vec2)
            # true labels
            c1, c2, _ = targets
            cls_pred = torch.cat([y1, y2], dim=0)
            cls_true = torch.cat([c1, c2], dim=0)

        ret = {
            "cls_pred": cls_pred,
            "cls_true": cls_true,
            "targets": targets
        }

        if self.l2_normalize:
            ret["emb_vecs"] = [l2_emb_vec1, l2_emb_vec2]
        else:
            ret["emb_vecs"] = [emb_vec1, emb_vec2]
        return ret

    def run(self, max_epochs=10):
        # make the scheduler first as it is different for different max_epochs
        train_ds = self.datasets['train']
        scheduler = CosineAnnealingLR(
            self.optimizer, T_max=len(train_ds) * max_epochs / self.batch_size,
            eta_min=1e-6)

        # ---------------
        # make engine
        # ---------------
        trainer = Engine(self.train_update)
        metrics = {
            "sim_acc": SiamSimAccuracy(
                margin=self.margin,
                output_transform=lambda x: (x['emb_vecs'], x['targets'])),
            "clsf_acc": Accuracy(
                output_transform=lambda x: (x['cls_pred'], x['cls_true']))
        }
        for name, metric in metrics.items():
            metric.attach(trainer, name)

        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {
            'con_loss': x['con_loss'],
            'clsf_loss': x['clsf_loss']
        })

        # --------------
        # callbacks
        # --------------
        # Scheduler step forwards
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED, lambda engine: scheduler.step())

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_acc(engine):
            epoch = engine.state.epoch
            metrics = engine.state.metrics
            sim_acc = metrics['sim_acc']
            clsf_acc = metrics['clsf_acc']
            pbar.log_message(
                "Epoch[{}] sim_acc: {:.2f}; clsf_acc {:.2f}"
                .format(epoch, sim_acc, clsf_acc))

        @trainer.on(Events.EPOCH_COMPLETED)
        def run_validation(engine):
            # prepare the evaluator
            evaluator = Engine(self.eval_inference)
            for name, metric in metrics.items():
                metric.attach(evaluator, name)
            ProgressBar(desc="Evaluating").attach(evaluator)
            evaluator.run(
                DataLoader(
                    self.datasets['siam_val'],
                    **self.loader_kwargs))
            avg_acc = evaluator.state.metrics['sim_acc']
            print(avg_acc)

        # start training
        trainer.run(
            DataLoader(
                self.datasets['siam_train'],
                **self.loader_kwargs, shuffle=True),
            max_epochs=max_epochs)
