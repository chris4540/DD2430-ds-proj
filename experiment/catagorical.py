"""
The simplest method to train a network for mapping img to embbeding space
"""
from .siamcos import SiameseCosDistanceWithCat
# utils
from ignite.contrib.handlers import ProgressBar
from collections import OrderedDict
from torch.utils.data import DataLoader
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
from ignite.engine import create_supervised_evaluator
from ignite.engine import Engine
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

    @property
    def evaluator(self):
        if not self._evaluator:
            # The evaluation metrics
            metrics = {
                "clsf_acc": Accuracy(),
                "clsf_loss": Loss(self.loss_fns['cross_entropy'])
            }
            # create the evaluator from the ignite factory function
            evaluator = create_supervised_evaluator(
                self.models['cnn_net'], metrics=metrics,
                device=self.device, non_blocking=self.pin_memory)

            # save down the evaluator
            self._evaluator = evaluator

        return self._evaluator

    def run(self):

        # make scheduler
        scheduler = self.scheduler
        # make trainer
        trainer = self.trainer
        # make evaluator
        evaluator = self.evaluator

        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {
            'clsf_loss': x['clsf_loss']
        })
        # --------------
        # callbacks
        # --------------
        # Scheduler step forwards
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED, lambda engine: scheduler.step())

        @trainer.on(Events.EPOCH_COMPLETED)
        def show_training_acc(engine):
            epoch = engine.state.epoch
            metrics = engine.state.metrics
            pbar.log_message(
                "Epoch[{}] clsf_loss:{clsf_loss:.2f}; clsf_acc: {clsf_acc:.2f}"
                .format(epoch, **metrics))

        def eval_callbacks(mode='val'):
            # prepare the evaluator
            ProgressBar(desc="Evalating on " + mode).attach(evaluator)
            ds = self.datasets[mode]
            evaluator.run(
                DataLoader(ds, **self.loader_kwargs))
            metrics = evaluator.state.metrics
            pbar.log_message(
                mode +
                " clsf_loss: {clsf_loss:.2f}; clsf_acc: {clsf_acc:.2f}"
                .format(**metrics))

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_to_csv(engine):

            log = OrderedDict()
            commom_cols = ["clsf_loss", "clsf_acc"]
            # Log training part
            log['epoch'] = engine.state.epoch
            for col in commom_cols:
                log["train_" + col] = engine.state.metrics[col]

            # log validation
            eval_callbacks(mode='val')
            for col in commom_cols:
                log["val_" + col] = evaluator.state.metrics[col]

            # log test
            eval_callbacks(mode='test')
            for col in commom_cols:
                log["test_" + col] = evaluator.state.metrics[col]
            # log learning rate
            log['lr'] = scheduler.get_lr()[0]
            # write down
            self.csv_logger.log_with_order(log)

        # save checkpoints
        to_save = {k: v for k, v in self.models.items() if k != "cnn_net"}
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            ModelCheckpoint(
                dirname=self.exp_folder, filename_prefix='',
                save_interval=1,
                n_saved=1e4, create_dir=True,
                save_as_state_dict=True,
                require_empty=False),
            to_save)

        # start training
        trainer.run(
            DataLoader(
                self.datasets['train'],
                **self.loader_kwargs, shuffle=True),
            max_epochs=self.hparams.epochs)
