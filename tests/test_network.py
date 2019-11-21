import torch
from network.resnet import ResidualEmbNetwork

def test_resnet16():
	input_img = torch.randn((1, 3, 224, 224))
	net = ResidualEmbNetwork()
	# net(input_img)
	# net(input_img).shape

