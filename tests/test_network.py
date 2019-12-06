import torch
import torch.nn as nn
from network.resnet import ResidualEmbNetwork
from network.clsf_net import ClassificationNet
# from network.resnet import ResidualNetwork
import torch.nn.functional as F


def test_resnet16_emb():
    batch_size = 5
    input_img = torch.randn((batch_size, 3, 224, 224))
    net = ResidualEmbNetwork()
    emb_vec = net(input_img)
    assert emb_vec.shape[0] == batch_size
    assert emb_vec.shape[1] > 0
    assert emb_vec.shape[1] == net.emb_dim


def test_resnet16():
    batch_size = 5
    n_cls = 10
    input_img = torch.randn((batch_size, 3, 224, 224))
    # net = ResidualNetwork(nb_classes=n_cls)
    emb_net = ResidualEmbNetwork()
    clsf_net = ClassificationNet(emb_net.emb_dim, nb_classes=n_cls)
    net = nn.Sequential(emb_net, clsf_net)
    out = net(input_img)

    target = torch.randint(n_cls, (batch_size,), dtype=torch.int64)
    loss = F.cross_entropy(out, target)
    loss.backward()
    assert loss > 0
