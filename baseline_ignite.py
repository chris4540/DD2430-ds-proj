"""
Implementation using pytorch ignite

Reference:
    https://github.com/pytorch/ignite/blob/v0.2.1/examples/mnist/mnist.py
    https://fam-taro.hatenablog.com/entry/2018/12/25/021346

TODO:
    resume from checkpoint (check statedict)
"""
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


class HyperParams:
    batch_size = 256
    lr = 5e-2
    log_interval = 50
    epochs = 5

    def __init__(self, hparams):
        pass

    def __repr__(self):
        pass

    def save_to_json(self):
        pass


class FashionMNISTConfig:
    root = "./"
    mean = 0.28604059698879553
    std = 0.35302424451492237


class BaselineFashionMNIST:
    """
    Test tube class for constructing embbeding space only with classifcation
    method
    """

    def __init__(self):
        pass

    @staticmethod
    def get_data_loaders():
        """
        Our target is to construct embbeding space. Therefore we use the "test set"
        as validation
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
        train_loader = DataLoader(
            train_ds, shuffle=True, batch_size=HyperParams.batch_size)
        val_loader = DataLoader(val_ds, shuffle=False,
                                batch_size=HyperParams.batch_size)
        return train_loader, val_loader

    def run(self, hparams):

        # Config
        cfg = HyperParams(hparams)

        # model
        model = SimpleCNN()
        self.model = model

        train_loader, val_loader = self.get_data_loaders()

        # device
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

        # learning rate scheduler
        scheduler = StepLR(optimizer=optimizer, step_size=2, gamma=0.1, last_epoch=-1)

        # loss function
        loss_fn = F.cross_entropy
        # trainer
        trainer = create_supervised_trainer(
            model, optimizer, loss_fn, device=device)

        evaluator = create_supervised_evaluator(model,
                                                metrics={'accuracy': Accuracy(),
                                                         'ce_loss': Loss(loss_fn)},
                                                device=device)

        desc = "ITERATION - loss: {:.2f}"
        pbar = tqdm(
            initial=0, leave=False, total=len(train_loader),
            desc=desc.format(0)
        )

        # checkpoints
        handler = ModelCheckpoint(dirname='./checkpoints', filename_prefix='sample',
                                  save_interval=2, n_saved=3, create_dir=True, save_as_state_dict=True)

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
        # trainer.add_event_handler(Events.I, lambda engine: lr_scheduler.step())
        @trainer.on(Events.EPOCH_COMPLETED)
        def take_scheduler_step(engine):
            scheduler.step()

            # Print out
            tqdm.write("Learning Rate - Epoch: {}  Learning Rate: {}"
                .format(engine.state.epoch, scheduler.get_lr()))



        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            iter = (engine.state.iteration - 1) % len(train_loader) + 1

            if iter % cfg.log_interval == 0:
                pbar.desc = desc.format(engine.state.output)
                pbar.update(cfg.log_interval)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            pbar.refresh()
            evaluator.run(train_loader)
            metrics = evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_ce_loss = metrics['ce_loss']
            tqdm.write(
                "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                .format(engine.state.epoch, avg_accuracy, avg_ce_loss)
            )

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_ce_loss = metrics['ce_loss']
            tqdm.write(
                "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                .format(engine.state.epoch, avg_accuracy, avg_ce_loss))

            pbar.n = pbar.last_print_n = 0

        trainer.run(train_loader, max_epochs=cfg.epochs)
        pbar.close()

    def save_model(self):
        pass


if __name__ == "__main__":
    hyper_params = dict()
    experiment = BaselineFashionMNIST()
    experiment.run(hyper_params)
