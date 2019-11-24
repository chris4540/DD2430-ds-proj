import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torchvision.datasets import FashionMNIST
from ignite.engine import Events
from ignite.engine import create_supervised_trainer
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy
from ignite.metrics import Loss
from ignite.handlers import ModelCheckpoint
from utils.loss import ContrastiveLoss
from utils.datasets import SiameseMNIST
from network.simple_cnn import SimpleConvEmbNet
from network.siamese import SiameseNet
from . import HyperParams
from .base import BaseTrainer
from config.fashion_mnist import FashionMNISTConfig
from utils import extract_embeddings
from utils.datasets import Siamesize
from .metrics import SimilarityAccuracy



class FashionMNISTConfig:
    root = "./"
    mean = 0.28604059698879553
    std = 0.35302424451492237


class BaselineFashionMNISTTrainer(BaseTrainer):
    """
    Test tube class for constructing embbeding space only with classifcation
    method
    """

    def __init__(self, log_interval=50, **kwargs):
        super().__init__(log_interval=log_interval)
        self.hparams = HyperParams(**kwargs)
        self.hparams.display()
        # self.hparams.save_to_txt('hp.txt')

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
        data_transform = Compose([
                ToTensor(),
                Normalize((cfg.mean,), (cfg.std,))
            ])

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
        # scheduler = self.scheduler
        loss_fn = self.loss_fn
        device = self.device
        eval_metrics = self.eval_metrics
        hparams = self.hparams
        # ----------------------------------
        # log_interval = self.log_cfg['interval']
        # desc = self.log_cfg['desc']
        # pbar = self.log_cfg['pbar']
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

        trainer.add_event_handler(
            Events.ITERATION_COMPLETED, self.log_training_loss)

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, self.log_training_results, **{
                'train_loader': train_loader,
                'evaluator': evaluator
            })

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, self.log_validation_results, **{
                'val_loader': val_loader,
                'evaluator': evaluator
            })

        trainer.run(train_loader, max_epochs=hparams.epochs)
        self.log_cfg['pbar'].close()

    def save_model(self):
        pass

class SiameseFashionMNISTTrainer(BaseTrainer):

    def __init__(self, log_interval=50, **kwargs):
        super().__init__(log_interval=log_interval)
        self.hparams = HyperParams(**kwargs)
        self.hparams.display()
        # self.hparams.save_to_txt('hp.txt')

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

        # ---------------------------------------------------
        # Returns pairs of images and target same/different
        # ---------------------------------------------------
        # from torch.utils.data import Subset
        # siamese_train_ds = Subset(siamese_train_ds, list(range(500)))
        siamese_train_ds = SiameseMNIST(train_ds)
        siamese_test_ds = SiameseMNIST(val_ds)

        # ----------------------------
        # Consturct loader
        # ----------------------------
        # self.val_loader = DataLoader(val_ds, shuffle=False,
        #                              batch_size=self.hparams.batch_size)

        batch_size = self.hparams.batch_size
        self.siamese_train_loader = DataLoader(
            siamese_train_ds, batch_size=batch_size, shuffle=True)
        self.siamese_val_loader = torch.utils.data.DataLoader(
            siamese_test_ds, batch_size=batch_size, shuffle=False)
        self.train_loader_len = len(self.siamese_train_loader)

    def prepare_exp_settings(self):
        # model
        emb_net = SimpleConvEmbNet()
        model = SiameseNet(emb_net)
        self.model = model

        # optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.hparams.lr)

        # learning rate scheduler
        self.scheduler = StepLR(
            optimizer=self.optimizer, step_size=2, gamma=0.1, last_epoch=-1)

        # loss function
        margin = 1.0
        self.loss_fn = ContrastiveLoss(margin)

        # evalution metrics
        self.eval_metrics = {
            'accuracy': SimilarityAccuracy(margin),
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
        train_loader = self.siamese_train_loader

        # trainer
        trainer = create_supervised_trainer(
            model, optimizer, loss_fn, device=device)

        evaluator = create_supervised_evaluator(
            model, metrics=eval_metrics, device=device)


        # learning rate
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, self.take_scheduler_step)

        trainer.add_event_handler(
            Events.ITERATION_COMPLETED, self.log_training_loss)

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, self.log_training_results, **{
                'train_loader': train_loader,
                'evaluator': evaluator
            })

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, self.log_validation_results, **{
                'val_loader': self.siamese_val_loader,
                'evaluator': evaluator
            })

        trainer.run(train_loader, max_epochs=hparams.epochs)
        pbar.close()

    # def map_val_ds_to_emb_space(self):
    #     #
    #     emb_net = self.model.emb_net
    #     loader = self.val_loader
    #     embeddings, labels = extract_embeddings(emb_net, loader)
    #     return embeddings, labels
