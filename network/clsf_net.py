import torch.nn as nn


class ClassificationNet(nn.Module):
    """
    A subnet as a part of resnet
    """

    def __init__(self, emb_dim=512, nb_classes=46):
        super().__init__()
        self.emb_dim = emb_dim
        self.nb_classes = nb_classes
        self._add_fully_connected_layer()

    def _add_fully_connected_layer(self):
        self.fc = nn.Sequential(
            nn.Linear(self.emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.nb_classes))

    def forward(self, emb_vecs):
        out = self.fc(emb_vecs)
        return out
