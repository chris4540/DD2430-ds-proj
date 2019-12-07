"""
The simplest method to train a network for mapping img to embbeding space
"""
from .siamcos import SiameseCosDistanceWithCat
# Networks
from network.resnet import ResidualEmbNetwork
from network.clsf_net import ClassificationNet

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
    def optimizer(self):
        if self._optimizer is None:
            cnn_net = self.models['cnn_net']
            optimizer = optim.Adam(
                cnn_net.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay)
            self._optimizer = optimizer

        return self._optimizer

    def train_update(self, engine, batch):
        """
        We define the training update function for engine use
        as we don't want to have a second pass through the training set

        See also:
            https://pytorch.org/ignite/quickstart.html#f1
        """
        pass
        # # alias
        # siam_net = self.models['siam_net']
        # optimizer = self.optimizer
        # con_loss_fn = self.loss_fns['contrastive']

        # siam_net.train()
        # optimizer.zero_grad()
        # x, targets = _prepare_batch(batch, device=self.device,
        #                             non_blocking=self.pin_memory)
        # emb_vec1, emb_vec2 = siam_net(x)

        # contras_loss = con_loss_fn((emb_vec1, emb_vec2), targets)

        # loss = contras_loss
        # loss.backward()
        # optimizer.step()

        # # contruct the return of the processing function of a engine
        # ret = {
        #     "con_loss": contras_loss.item(),
        #     "targets": targets,
        #     "emb_vecs": [emb_vec1, emb_vec2]
        # }

        # return ret
