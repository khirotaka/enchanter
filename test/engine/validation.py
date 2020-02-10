import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms

import enchanter


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 28*28)
        x = self.fc(x)
        return x


def main():
    train_ds = MNIST("../data", train=True, download=False, transform=transforms.ToTensor())
    val_ds = MNIST("../data", train=True, download=False, transform=transforms.ToTensor())

    val_size = 0.1
    n_trains = len(train_ds)
    indices = list(range(n_trains))
    splits = int(np.floor(val_size * n_trains))
    train_idx, val_idx = indices[splits:], indices[:splits]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    model = Network()

    runner = enchanter.ClassificationRunner(model, nn.CrossEntropyLoss(), optim.Adam, {"lr": 0.001})
    runner.train(train_ds, 1, 64, sampler=train_sampler, validation={
        "dataset": val_ds,
        "config": {
            "sampler": val_sampler
        }
    })


if __name__ == '__main__':
    main()
