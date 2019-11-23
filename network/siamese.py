import torch
import torch.nn as nn

class SiameseNet(nn.Module):
    def __init__(self, emb_net):
        super().__init__()
        self._emb_net = emb_net

    def forward(self, x):
        x1, x2 = x
        out1 = self._emb_net(x1)
        out2 = self._emb_net(x2)
        return out1, out2

    @property
    def emb_net(self):
        return self._emb_net
