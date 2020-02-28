import torch.nn as nn
import enchanter.addon as addon


class MNIST(nn.Module):
    """
    MNIST 用のCNN
    """
    def __init__(self):
        super(MNIST, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            addon.Swish(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            addon.Swish(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*5*5, 512),
            addon.Swish(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, 64*5*5)
        out = self.fc(out)
        return out
