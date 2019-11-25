import torchvision.models as models
import torch.nn as nn


class ResidualEmbNetwork(nn.Module):
    """
    The embbeding network
    """

    def __init__(self, depth=18):
        super().__init__()

        depth_opts = [18, 34, 50]
        if depth not in depth_opts:
            raise ValueError("Supports only resnet 18, 34 and 50")

        self.depth = depth
        if depth == 18:
            net = models.resnet18(pretrained=True)
            self.emb_dim = 512
        elif depth == 34:
            net = models.resnet34(pretrained=True)
            self.emb_dim = 512
        elif depth == 50:
            net = models.resnet50(pretrained=True)
            self.emb_dim = 2048

        # drop the last fully connected layer
        self.net = nn.Sequential(*(list(net.children())[:-1]))
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = self.net(x)
        out = self.flatten(out)
        return out


class ResidualNetwork(nn.Module):

    def __init__(self, depth=18, nb_classes=46):
        super().__init__()
        self._emb_net = ResidualEmbNetwork(depth)
        self.fc = nn.Sequential(
            nn.Linear(self._emb_net.emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, nb_classes))

    def forward(self, x):
        emb_vec = self._emb_net(x)
        out = self.fc(emb_vec)
        return out

    @property
    def emb_net(self):
        return self._emb_net
