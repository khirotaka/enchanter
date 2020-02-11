import comet_ml
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from sklearn.metrics import accuracy_score
from torch.utils.data import SubsetRandomSampler

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


class CustomRunner(enchanter.ClassificationRunner):
    def validate(self, data: torch.Tensor, target: torch.Tensor):
        results = super(CustomRunner, self).validate(data, target)
        results["accuracy_score"] = accuracy_score(target.cpu().numpy(), self.predict(data))

        return results


def main():
    PATH = "../test/data/"
    train_ds = MNIST(PATH, train=True, transform=transforms.ToTensor())
    test_ds = MNIST(PATH, train=False, transform=transforms.ToTensor())
    val_ds = MNIST(PATH, train=True, download=True, transform=transforms.ToTensor())
    pin_memory = torch.cuda.is_available()

    val_size = 0.1
    n_trains = len(train_ds)
    indices = list(range(n_trains))
    splits = int(np.floor(val_size * n_trains))
    train_idx, val_idx = indices[splits:], indices[:splits]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    runner = CustomRunner(
        Network(), nn.CrossEntropyLoss(), optim.Adam, {"lr": 0.001},
        experiment=comet_ml.Experiment(project_name="testflight")
    )
    runner.train(
        train_ds, epochs=2, batch_size=128,
        checkpoint=PATH+"checkpoint/",
        num_workers=1,
        sampler=train_sampler,
        pin_memory=pin_memory,
        shuffle=False,
        validation={
            "dataset": val_ds,
            "config": {
                "num_workers": 1,
                "sampler": val_sampler,
                "pin_memory": pin_memory
            }
        }
    )
    loss, accuracy = runner.evaluate(test_ds, batch_size=64)
    print(loss)
    print(accuracy)


if __name__ == '__main__':
    main()
