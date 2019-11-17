import torch
import torch.nn as nn

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x):
        x1, x2 = x
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2
