import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms

import enchanter
import enchanter.addon as addon


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
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


def main():
    train_ds = MNIST("../test/data/", train=True, transform=transforms.ToTensor())
    test_ds = MNIST("../test/data/", train=False, transform=transforms.ToTensor())

    runner = enchanter.ClassificationRunner(Network(), nn.CrossEntropyLoss(), optim.Adam, {"lr": 0.001})
    runner.train(train_ds, epochs=1, batch_size=64, shuffle=True, verbose=False)
    loss, accuracy = runner.evaluate(test_ds, batch_size=64)
    print(loss)
    print(accuracy)


if __name__ == '__main__':
    main()
