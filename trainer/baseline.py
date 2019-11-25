import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from . import HyperParams
from .base import BaseTrainer
from .loss import ContrastiveLoss
# from .metrics import SimilarityAccuracy
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from ignite.engine import Events
from ignite.engine import create_supervised_trainer
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Loss
from ignite.metrics import Accuracy
from ignite.handlers import ModelCheckpoint
from network.siamese import SiameseNet
from network.resnet import ResidualNetwork
from utils.datasets import DeepFashionDataset
from utils.datasets import Siamesize
from torch.utils.data import Subset
from utils import extract_embeddings
from config.deep_fashion import DeepFashionConfig as cfg
from tqdm import tqdm
from annoy import AnnoyIndex


class ClassificationTrainer(BaseTrainer):

    _debug = False

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

        self.train_ds = DeepFashionDataset(
            cfg.root_dir, 'train', transform=trans)
        self.val_ds = DeepFashionDataset(
            cfg.root_dir, 'val', transform=trans)

        # ----------------------------
        # Consturct data loader
        # ----------------------------
        loader_kwargs = {
            'pin_memory': True,
            'batch_size': self.hparams.batch_size,
            'num_workers': os.cpu_count(),
        }
        self.train_loader = DataLoader(
            self.train_ds, shuffle=True, **loader_kwargs)
        self.val_loader = DataLoader(
            self.val_ds, shuffle=False, **loader_kwargs)

    def prepare_exp_settings(self):
        # model
        # emb_net = ResidualEmbNetwork()
        # model = SiameseNet(emb_net)
        model = ResidualNetwork(nb_classes=15)
        self.model = model

        # optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.hparams.lr)

        # learning rate scheduler
        self.scheduler = StepLR(
            optimizer=self.optimizer, step_size=2, gamma=0.5, last_epoch=-1)

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
        loss_fn = self.loss_fn
        device = self.device
        eval_metrics = self.eval_metrics
        hparams = self.hparams
        #
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

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, self.log_training_results, **{
                'train_loader': train_loader,
                'evaluator': evaluator
            })

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, self.log_validation_results, **{
                'val_loader': self.val_loader,
                'evaluator': evaluator
            })
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, self.log_topk_retrieval_acc)

        trainer.run(train_loader, max_epochs=hparams.epochs)
        pbar.close()

    # top k retrival acc
    def log_topk_retrieval_acc(self, engine):
        """
        For tracking the performance during training top K Precision
        """
        loader_kwargs = {
            'pin_memory': True,
            'num_workers': os.cpu_count(),
            'batch_size': 100
        }
        train_loader = DataLoader(self.train_ds, **loader_kwargs)
        val_loader = DataLoader(self.val_ds, **loader_kwargs)

        # ----------------------------------
        train_embs, train_labels = extract_embeddings(self.model, train_loader)
        val_embs, val_labels = extract_embeddings(self.model, val_loader)
        emb_dim = train_embs.shape[1]
        # ----------------------------------
        t = AnnoyIndex(emb_dim, metric='euclidean')
        n_trees = 100
        for i, emb_vec in enumerate(train_embs):
            t.add_item(i, emb_vec)
        # build a forest of trees
        tqdm.write("Building ANN forest...")
        t.build(n_trees)
        # ----------------------------------
        top_k_corrects = dict()
        # Meassure Prec@[5, 10, 20, 30]
        for i, emb_vec in enumerate(val_embs):
            correct_cls = val_labels[i]
            for k in [5, 10, 20, 30]:
                idx = t.get_nns_by_vector(emb_vec, k)
                top_k_classes = train_labels[idx]
                correct = np.sum(top_k_classes == correct_cls)
                accum_corr = top_k_corrects.get(k, 0)
                top_k_corrects[k] = accum_corr + correct
        # -------------------------------------------------
        # calculate back the acc
        top_k_acc = dict()
        for k in [5, 10, 20, 30]:
            top_k_acc[k] = top_k_corrects[k] / k / val_embs.shape[0]

        tqdm.write(
            "Top K Retrieval Results - Epoch: {}  Avg top-k accuracy:"
            .format(engine.state.epoch)
        )

        for k in [5, 10, 20, 30]:
            tqdm.write("  Prec@{} = {:.2f}".format(k, top_k_acc[k]))
