from comet_ml import Experiment

import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

import enchanter.wrappers as wrappers
import models


def main():
    experiment = Experiment()

    train_ds = MNIST("../tests/data/", train=True, transform=transforms.ToTensor())
    test_ds = MNIST("../tests/data/", train=False, transform=transforms.ToTensor())
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = models.MNIST()
    optimizer = optim.Adam(model.parameters())
    runner = wrappers.ClassificationRunner(
        model,
        optimizer,
        nn.CrossEntropyLoss(),
        experiment,
        scheduler=optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    )
    runner.add_loader(train_loader, "train").add_loader(test_loader, "test").train_config(epochs=1)

    runner.run(verbose=True)


if __name__ == '__main__':
    main()
