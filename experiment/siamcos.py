import os
from network.resnet import ResidualEmbNetwork
from network.siamese import SiameseNet
from network.clsf_net import ClassificationNet
# datasets
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from utils.datasets import Siamesize
from utils.datasets import DeepFashionDataset
from config.deep_fashion import DeepFashionConfig as cfg

class SiameseCosDistance:
    """
    Example:
    >> exp = SiameseCosDistance()
    >> exp.run(max_epochs=10)
    """

    # opts to make small dataset for overfitting as debug purpose
    _debug = False

    #
    _models = None
    _datasets = None

    def __init__(self):
        self.batch_size = 128
        self.loader_kwargs = {
            'pin_memory': True,
            'batch_size': self.batch_size,
            'num_workers': os.cpu_count()
        }
        self.device = 'cpu'


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
                train_samples = np.random.choice(len(train_ds), 300, replace=False)
                val_samples = np.random.choice(len(val_ds), 100, replace=False)
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

    # @property
    # def scheduler(self):
    #     if not self._scheduler:
    #         T_max = len(self.datasets['train']) * max_epochs / batch_size
    #         self._scheduler = CosineAnnealingLR(
    #             self.optimizer, T_max=len(train_ds) * max_epochs / batch_size,
    #             eta_min=1e-6)
    #     return self._scheduler

    @property
    def loss_fns(self):
        if not self._loss_fns:
            self._loss_fns = {
                'contrastive': ContrastiveLoss(margin=self.margin, average=True),
                'cross_entropy': CrossEntropyLoss()
            }
        return self._loss_fns


    @property
    def hparams(self):
        return self._hparams


    def run(self, max_epochs=10):
        pass
