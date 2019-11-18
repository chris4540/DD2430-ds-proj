"""
Implementation using pytorch ignite

Reference:
    https://github.com/pytorch/ignite/blob/v0.2.1/examples/mnist/mnist.py
    https://fam-taro.hatenablog.com/entry/2018/12/25/021346

TODO:
    resume from checkpoint (check statedict)
"""
from . import HyperParams
from .base import BaseTrainingMethod
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torchvision.datasets import FashionMNIST
from network.simple_cnn import SimpleCNN
from ignite.engine import Events
from ignite.engine import create_supervised_trainer
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy
from ignite.metrics import Loss
from ignite.handlers import ModelCheckpoint
# from argparse import ArgumentParser
from tqdm import tqdm


class FashionMNISTConfig:
    root = "./"
    mean = 0.28604059698879553
    std = 0.35302424451492237


class BaselineFashionMNIST(BaseTrainingMethod):
    """
    Test tube class for constructing embbeding space only with classifcation
    method
    """

    def __init__(self, log_interval=50, **kwargs):
        super().__init__(log_interval=log_interval)
        self.hparams = HyperParams(**kwargs)
        self.hparams.display()

        # check if cpu or gpu
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def prepare_data_loaders(self):
        """
        Our target is to construct embbeding space.
        Therefore we use the "test set" as validation
        """
        # alias
        cfg = FashionMNISTConfig

        # data transform
        data_transform = Compose(
            [ToTensor(), Normalize((cfg.mean,), (cfg.std,))])

        # ----------------------------
        # Consturct data loader
        # ----------------------------
        ds_kwargs = {
            'root': cfg.root,
            'transform': data_transform
        }
        train_ds = FashionMNIST(train=True, download=True, **ds_kwargs)
        val_ds = FashionMNIST(train=False, download=False, **ds_kwargs)
        # ----------------------------
        # Consturct loader
        # ----------------------------
        self.train_loader = DataLoader(
            train_ds, shuffle=True, batch_size=HyperParams.batch_size)
        self.val_loader = DataLoader(val_ds, shuffle=False,
                                     batch_size=HyperParams.batch_size)

    def prepare_exp_settings(self):
        # model
        self.model = SimpleCNN()

        # optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.hparams.lr)

        # learning rate scheduler
        self.scheduler = StepLR(
            optimizer=self.optimizer, step_size=2, gamma=0.1, last_epoch=-1)

        # loss function
        self.loss_fn = F.cross_entropy

        # evalution metrics
        self.eval_metrics = {
            'accuracy': Accuracy(),
            'loss': Loss(self.loss_fn)
        }

    # ------------------------------------------------------------------
    def run(self):
        self.prepare_before_run()
        # ----------------------------------
        # Alias
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        loss_fn = self.loss_fn
        device = self.device
        eval_metrics = self.eval_metrics
        hparams = self.hparams
        # ----------------------------------
        log_interval = self.log_cfg['interval']
        desc = self.log_cfg['desc']
        pbar = self.log_cfg['pbar']
        # ----------------------------------
        # Special alias
        train_loader = self.train_loader
        val_loader = self.val_loader

        # trainer
        trainer = create_supervised_trainer(
            model, optimizer, loss_fn, device=device)

        evaluator = create_supervised_evaluator(
            model, metrics=eval_metrics, device=device)

        # learning rate
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, self.take_scheduler_step)

        # trainer.add_event_handler(
        #     Events.ITERATION_COMPLETED, self.log_training_loss)

        # trainer.add_event_handler(
        #     Events.EPOCH_COMPLETED, self.log_training_results, **{
        #         'train_loader': train_loader,
        #         'evaluator': evaluator
        #     })

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            iter = (engine.state.iteration - 1) % len(train_loader) + 1

            if iter % log_interval == 0:
                pbar.desc = desc.format(engine.state.output)
                pbar.update(log_interval)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            pbar.refresh()
            evaluator.run(train_loader)
            metrics = evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_loss = metrics['loss']
            tqdm.write(
                "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                .format(engine.state.epoch, avg_accuracy, avg_loss)
            )

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_loss = metrics['loss']
            tqdm.write(
                "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                .format(engine.state.epoch, avg_accuracy, avg_loss))

            pbar.n = pbar.last_print_n = 0

        trainer.run(train_loader, max_epochs=hparams.epochs)
        pbar.close()

    def save_model(self):
        pass
