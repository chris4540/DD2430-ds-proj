import os
import numpy as np
import torch
import torch.optim as optim
from . import HyperParams
from .base import BaseTrainer
from .loss import ContrastiveLoss
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from ignite.engine import Events
from ignite.engine import create_supervised_trainer
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy
from ignite.metrics import Loss
from ignite.handlers import ModelCheckpoint
from network.siamese import SiameseNet
from network.resnet import ResidualEmbNetwork
from utils.datasets import DeepFashionDataset
from utils.datasets import Siamesize
from torch.utils.data import Subset
from utils import extract_embeddings
from config.deep_fashion import DeepFashionConfig as cfg


class SiameseTrainer(BaseTrainer):

    def __init__(self, exp_folder, log_interval=50, **kwargs):
        super().__init__(exp_folder=exp_folder, log_interval=log_interval)
        self.hparams = HyperParams(**kwargs)
        self.hparams.display()

        self.hparams.save_to_txt(self.exp_folder / 'hparams.txt')
        self.hparams.save_to_json(self.exp_folder / 'hparams.json')

        # check if cpu or gpu
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def prepare_data_loaders(self):
        """
        Our target is to construct embbeding space. Therefore we use the "test set"
        as validation
        """

        # data transform
        trans = Compose(
            [
                Resize(cfg.sizes),
                ToTensor(),
                Normalize(cfg.mean, cfg.std),

            ])

        # ----------------------------
        # Consturct data loader
        # ----------------------------
        self.train_ds = DeepFashionDataset(
            cfg.root_dir, 'train', transform=trans)
        # ---------------------------------------------------
        # Returns pairs of images and target same/different
        # ---------------------------------------------------
        siamese_train_ds = Siamesize(self.train_ds)

        # self.train_loader = DataLoader(
        #     siamese_train_ds, batch_size=self.hparams.batch_size, pin_memory=True, num_workers=os.cpu_count())
        self.train_loader = DataLoader(
            siamese_train_ds, batch_size=self.hparams.batch_size)

    def prepare_exp_settings(self):
        # model
        emb_net = ResidualEmbNetwork()
        model = SiameseNet(emb_net)
        self.model = model

        # optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.hparams.lr)

        # learning rate scheduler
        self.scheduler = StepLR(
            optimizer=self.optimizer, step_size=5, gamma=0.1, last_epoch=-1)

        # loss function
        margin = 1.0
        self.loss_fn = ContrastiveLoss(margin)

        # evalution metrics
        self.eval_metrics = {
            'loss': Loss(self.loss_fn)
        }
    # ------------------------------------------------------------------

    def run(self):
        self.prepare_before_run()
        # ----------------------------------
        # Alias
        model = self.model
        optimizer = self.optimizer
        # scheduler = self.scheduler
        loss_fn = self.loss_fn
        device = self.device
        eval_metrics = self.eval_metrics
        hparams = self.hparams
        # ----------------------------------
        # log_interval = self.log_cfg['interval']
        # desc = self.log_cfg['desc']
        pbar = self.log_cfg['pbar']
        # ----------------------------------
        # Special alias
        train_loader = self.train_loader

        # trainer
        trainer = create_supervised_trainer(
            model, optimizer, loss_fn, device=device)

        evaluator = create_supervised_evaluator(
            model, metrics=eval_metrics, device=device)

        # checkpoints
        handler = ModelCheckpoint(
            dirname=str(self.exp_folder / 'chkptr'),
            filename_prefix='',
            save_interval=1,
            n_saved=hparams.epochs,
            create_dir=True,
            save_as_state_dict=True,
            require_empty=False)

        # -------------------
        # Callbacks / Events
        # -------------------

        # check point
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, handler, {
                'model': model,
                "optimizer": optimizer,
            })

        # learning rate
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, self.take_scheduler_step)

        trainer.add_event_handler(
            Events.ITERATION_COMPLETED, self.log_training_loss)

        trainer.run(train_loader, max_epochs=hparams.epochs)
        pbar.close()

    def save_model(self):
        torch.save(self.model.state_dict(), 'siamese_resnet18.pth')

    def map_train_ds_to_emb_space(self):
        #
        emb_net = self.model.emb_net
        # subset
        n_samples = 28000
        sel_idx = np.random.choice(
            list(range(len(self.train_ds))),
            n_samples, replace=False)

        assert len(set(sel_idx)) == n_samples

        ds = Subset(self.train_ds, sel_idx)
        loader = DataLoader(
            ds, batch_size=self.hparams.batch_size, pin_memory=True, num_workers=2)
        embeddings, labels = extract_embeddings(emb_net, loader)
        return embeddings, labels
