from .siamcos import SiameseCosDistanceWithCat
# Networks
from network.resnet import ResidualEmbNetwork
from network.siamese import SiameseNet

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
        clsf_net.train()
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
            "loss": loss.item(),
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
                "loss": Average(output_transform=lambda x: x["loss"]),
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
                "sim_acc": SiamSimAccuracy(
                    margin=self.margin,
                    output_transform=lambda x: (x['emb_vecs'], x['targets'])),
                "con_loss": Loss(
                    self.loss_fns['contrastive'],
                    output_transform=lambda x: (x['emb_vecs'], x['targets']))
            }
            # create the evaluator from the ignite factory function
            evaluator = create_supervised_evaluator(
                self.models['siam_net'], metrics=metrics,
                device=self.device, non_blocking=self.pin_memory)

            # save down the evaluator
            self._evaluator = evaluator

        return self._evaluator

