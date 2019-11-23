import torch
from network.resnet import ResidualEmbNetwork

def test_resnet16():
	batch_size = 5
	input_img = torch.randn((batch_size, 3, 224, 224))
	net = ResidualEmbNetwork()
	emb_vec = net(input_img)
	print(emb_vec.shape)
	assert emb_vec.shape[0] == batch_size
	assert emb_vec.shape[1] > 0
	assert emb_vec.shape[1] == net.emb_dim


