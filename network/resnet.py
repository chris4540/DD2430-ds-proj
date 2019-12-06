import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x, l2_normalize=False):
        out = self.net(x)
        out = self.flatten(out)

        if l2_normalize:
            out = F.normalize(out, p=2, dim=1)
        return out


# class ResidualNetwork(nn.Module):

#     def __init__(self, depth=18, nb_classes=46, emb_net=None):
#         super().__init__()
#         if emb_net:
#             # copy the reference
#             self._emb_net = emb_net
#             self.depth = emb_net.depth
#         else:
#             self.depth = depth
#             self._emb_net = ResidualEmbNetwork(depth)

#         self.nb_classes = nb_classes
#         self._add_fully_connected_layer()

#     def _add_fully_connected_layer(self):
#         self.fc = nn.Sequential(
#             nn.Linear(self._emb_net.emb_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, self.nb_classes))

#     def forward(self, x):
#         emb_vecs = self._emb_net(x)
#         out = self.foward_from_emb_vec(emb_vecs)
#         return out

#     def foward_from_emb_vec(self, emb_vecs):
#         out = self.fc(emb_vecs)
#         return out

#     @property
#     def emb_net(self):
#         return self._emb_net
