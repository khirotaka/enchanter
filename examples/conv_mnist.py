from comet_ml import Experiment

import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

import enchanter.tasks as tasks
from enchanter.callbacks import EarlyStopping
import models


def main():
    experiment = Experiment()

    train_ds = MNIST("../tests/data/", download=True, train=True, transform=transforms.ToTensor())
    test_ds = MNIST("../tests/data/", download=True, train=False, transform=transforms.ToTensor())
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = models.MNIST()
    optimizer = optim.Adam(model.parameters())
    runner = tasks.ClassificationRunner(
        model,
        optimizer,
        nn.CrossEntropyLoss(),
        experiment,
        scheduler=[optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)],
        early_stop=EarlyStopping("train_avg_loss", min_delta=0.1, patience=1)
    )
    runner.add_loader("train", train_loader).add_loader("test", test_loader)\
        .train_config(
        epochs=5,
        checkpoint_path="./checkpoints",
        monitor="validate_avg_acc >= 0.75"
    )

    runner.run(verbose=True)


if __name__ == '__main__':
    main()
