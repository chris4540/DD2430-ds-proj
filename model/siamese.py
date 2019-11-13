"""
MIT License

Copyright (c) 2019 Chun Hung Lin, Vivek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
"""
Requirement:
1. Enable to put a model with certain architecture (Abstract class)
2. Share weightings
3. Implmenet loss functions
4. Simple dataset to show it works (minst/other dataset which run fast and light)

Ref:

# Keras
https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py
https://github.com/sugi-chan/shoes_siamese/blob/master/Siamese-clothes.ipynb
https://github.com/PlabanM1/FashionNet/blob/master/FashionNet.ipynb

# PyTorch
https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch/blob/master/Siamese-networks-medium.ipynb
https://github.com/fangpin/siamese-pytorch/blob/5543f1e844964b116dc9d347a5eb164c6a7afe6d/model.py#L6
https://github.com/adambielski/siamese-triplet/blob/master/Experiments_FashionMNIST.ipynb
Refer to my notebooks for required changes to siamese to allign them to the paper
"""
import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms

if __name__ == '__main__':
	pass
