import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    This is a simplest network to fit with fashion minst dataset.
    This network is to fasten the development time for the cycle:
        - coding
        - testing
        - visualization
    """

    def __init__(self, last_layer='logits'):

        super().__init__()

        self.last_layer = last_layer

        self.convnet = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 50),
        )

        #
        self.clsf = nn.Sequential(
            nn.ReLU(),
            nn.Linear(50, 10))

    def forward(self, x):
        """
        Return a logits vector
        """
        out = self.fwd_to_emb_layer(x)

        if self.last_layer == 'logits':
            out = self.clsf(out)
        return out

    def fwd_to_emb_layer(self, x):
        out = self.convnet(x)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out


if __name__ == '__main__':

    fashion_minst_shape = (1, 28, 28)

    # random input
    input_img = torch.randn((1, *fashion_minst_shape))

    # model
    model = SimpleCNN()
    output = model(input_img)
    print(output.shape)
