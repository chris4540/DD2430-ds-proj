"""
Implementation using pytorch ignite

Reference:
    https://github.com/pytorch/ignite/blob/v0.2.1/examples/mnist/mnist.py
    https://fam-taro.hatenablog.com/entry/2018/12/25/021346

TODO:
    resume from checkpoint (check statedict)
"""
from . import HyperParams
from .base import BaseTrainer
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torchvision.datasets import FashionMNIST
from network.simple_cnn import SimpleCNN
from ignite.engine import Events
from ignite.engine import create_supervised_trainer
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy
from ignite.metrics import Loss
from ignite.handlers import ModelCheckpoint
from tqdm import tqdm
