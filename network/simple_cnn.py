import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleConvEmbNet(nn.Module):
    """
    This is a simplest network to fit with fashion minst dataset.
    This network is to fasten the development time for the cycle:
        - coding
        - testing
        - visualization
    """

    def __init__(self, emb_size=50):
        super().__init__()
        self.emb_size = emb_size

        # The convolution block
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        # flatten layer
        self.flatten = nn.Flatten()

        # Fully connected block, expect the last layer to reduce to num class
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, self.emb_size),
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.flatten(out)
        out = self.fc(out)
        ret = out
        return ret


class SimpleCNN(nn.Module):
    """
    Using composition instead of inherit
    """

    def __init__(self, emb_size=50, n_classes=10):
        super().__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes

        self._emb_net = SimpleConvEmbNet(emb_size)
        #
        self.clsf = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        # Forward to embbeding space
        emb_vec = self._emb_net(x)

        # forward to logit vector
        logit_vec = self.clsf(emb_vec)
        ret = logit_vec
        return ret

    @property
    def emb_net(self):
        return self._emb_net


if __name__ == '__main__':

    fashion_minst_shape = (1, 28, 28)

    # random input
    input_img = torch.randn((1, *fashion_minst_shape))

    # model
    model = SimpleCNN()
    output = model(input_img)
    print(output.shape)
