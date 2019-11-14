import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple CNN
    Notes:
    The 
    """
    def __init__(self):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 32, 5), 
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5), 
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.PReLU(),
            nn.Linear(256, 256))

        # 
        self.clsf = nn.Sequential(
            nn.PReLU(),
            nn.Linear(256, 10))

    def forward(self, x):
        """
        Return a logits vector
        """
        out = self.forward_to_embbeding(x)
        out = self.clsf(out)
        return out

    def forward_to_embbeding(self, x):
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