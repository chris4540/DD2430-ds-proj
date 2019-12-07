"""
The simplest method to train a network for mapping img to embbeding space
"""
from .siamcos import SiameseCosDistanceWithCat
# Networks
import torch.nn as nn
from network.resnet import ResidualEmbNetwork
from network.clsf_net import ClassificationNet
# Optimizer
import torch.optim as optim
# metrics
from ignite.metrics import Accuracy
from ignite.metrics import Average
from ignite.metrics import Loss
from utils.metrics import SiamSimAccuracy
# training
from ignite.engine import _prepare_batch
from ignite.engine.engine import Engine
from ignite.engine import Events
# handler
from ignite.handlers import ModelCheckpoint


class CatClassification(SiameseCosDistanceWithCat):

    # Modifications:
    #   - Models
    #   - Optimizer

    @property
    def models(self):
        if self._models is None:
            emb_net = ResidualEmbNetwork()
            clsf_net = ClassificationNet(emb_net.emb_dim, nb_classes=15)
            # simply sequentially join the two networks
            cnn_net = nn.Sequential(emb_net, clsf_net)
            self._models = {
                "emb_net": emb_net,
                "clsf_net": clsf_net,
                "cnn_net": cnn_net
            }

            for model in self.models.values():
                model.to(self.device)

        return self._models

    @property
    def model_params(self):
        models = self.models
        cnn_net = models['cnn_net']
        ret = cnn_net.parameters()
        return ret

    def train_update(self, engine, batch):
        """
        We define the training update function for engine use
        as we don't want to have a second pass through the training set

        See also:
            https://pytorch.org/ignite/quickstart.html#f1
        """

        # alias
        cnn_net = self.models['cnn_net']
        optimizer = self.optimizer
        loss_fn = self.loss_fns['cross_entropy']

        cnn_net.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch, device=self.device,
                              non_blocking=self.pin_memory)
        y_pred = cnn_net(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        # contruct the return of the processing function of a engine
        ret = {
            "clsf_loss": loss.item(),
            "cls_pred": y_pred,
            "cls_true": y,
        }

        return ret

    @property
    def trainer(self):
        if not self._trainer:
            trainer = Engine(self.train_update)
            metrics = {
                "clsf_acc": Accuracy(
                    output_transform=lambda x: (x['cls_pred'], x['cls_true'])),
                "clsf_loss": Average(output_transform=lambda x: x["clsf_loss"])
            }
            for name, metric in metrics.items():
                metric.attach(trainer, name)
            self._trainer = trainer
            self.train_metrics = metrics

        return self._trainer

    def run(self):

        # make scheduler
        scheduler = self.scheduler
        # make trainer
        trainer = self.trainer
        # make evaluator
        evaluator = self.evaluator
