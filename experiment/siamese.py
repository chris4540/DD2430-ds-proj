from .siamcos import SiameseCosDistanceWithCat
# utils
from ignite.contrib.handlers import ProgressBar
from collections import OrderedDict
# datasets
from torch.utils.data import DataLoader
# Networks
from network.resnet import ResidualEmbNetwork
from network.siamese import SiameseNet
# Loss
from utils.loss import ContrastiveLoss
# Factory function
from ignite.engine import create_supervised_evaluator
# training
from ignite.engine import _prepare_batch
from ignite.engine.engine import Engine
from ignite.engine import Events
# metrics
from ignite.metrics import Average
from ignite.metrics import Loss
from utils.metrics import SiamSimAccuracy
# Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
# handler
from ignite.handlers import ModelCheckpoint


class Siamese(SiameseCosDistanceWithCat):

    # Modifications:
    #   - Models
    #   - Optimizer

    @property
    def models(self):
        if self._models is None:
            emb_net = ResidualEmbNetwork()
            siamese_net = SiameseNet(emb_net)
            self._models = {
                "emb_net": emb_net,
                "siam_net": siamese_net,
            }

            for model in self.models.values():
                model.to(self.device)

        return self._models

    @property
    def optimizer(self):
        if self._optimizer is None:
            models = self.models
            siam_net = models['siam_net']
            optimizer = optim.Adam(
                siam_net.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay)
            self._optimizer = optimizer

        return self._optimizer

    def train_update(self, engine, batch):
        # alias
        siam_net = self.models['siam_net']
        optimizer = self.optimizer
        con_loss_fn = self.loss_fns['contrastive']

        siam_net.train()
        optimizer.zero_grad()
        x, targets = _prepare_batch(batch, device=self.device,
                                    non_blocking=self.pin_memory)
        emb_vec1, emb_vec2 = siam_net(x)

        contras_loss = con_loss_fn((emb_vec1, emb_vec2), targets)

        loss = contras_loss
        loss.backward()
        optimizer.step()

        # contruct the return of the processing function of a engine
        ret = {
            "con_loss": contras_loss.item(),
            "targets": targets,
            "emb_vecs": [emb_vec1, emb_vec2]
        }

        return ret

    @property
    def trainer(self):
        if not self._trainer:
            trainer = Engine(self.train_update)
            metrics = {
                "sim_acc": SiamSimAccuracy(
                    margin=self.margin,
                    output_transform=lambda x: (x['emb_vecs'], x['targets'])),
                "con_loss": Average(output_transform=lambda x: x["con_loss"]),
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
                "sim_acc": SiamSimAccuracy(margin=self.margin),
                "con_loss": Loss(self.loss_fns['contrastive'])
            }
            # create the evaluator from the ignite factory function
            evaluator = create_supervised_evaluator(
                self.models['siam_net'], metrics=metrics,
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
            'con_loss': x['con_loss'],
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
                "Epoch[{}] sim_acc: {sim_acc:.2f};"
                .format(epoch, **metrics))

        def eval_callbacks(mode='val'):
            # prepare the evaluator
            ProgressBar(desc="Evalating on " + mode).attach(evaluator)
            ds = self.datasets['siam_' + mode]
            evaluator.run(
                DataLoader(ds, **self.loader_kwargs))
            metrics = evaluator.state.metrics
            pbar.log_message(
                mode +
                " sim_acc: {sim_acc:.2f}; con_loss: {con_loss:.2f};"
                .format(**metrics))

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_to_csv(engine):

            log = OrderedDict()
            commom_cols = ["con_loss", "sim_acc"]
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
        to_save = {k: v for k, v in self.models.items() if k != "siam_net"}
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
                self.datasets['siam_train'],
                **self.loader_kwargs, shuffle=True),
            max_epochs=self.hparams.epochs)
