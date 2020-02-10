import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import enchanter
import enchanter.ensemble as ensemble


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
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
    train_ds = MNIST("../../data", train=True, download=False, transform=ToTensor())
    test_ds = MNIST("../../data", train=False, download=False, transform=ToTensor())

    runner1 = enchanter.ClassificationRunner(Model(), nn.CrossEntropyLoss(), optim.Adam, optim_config={"lr": 0.001})
    runner2 = enchanter.ClassificationRunner(Model(), nn.CrossEntropyLoss(), optim.Adam, optim_config={"lr": 0.002})
    runner3 = enchanter.ClassificationRunner(Model(), nn.CrossEntropyLoss(), optim.Adam, optim_config={"lr": 0.003})

    soft = ensemble.SoftEnsemble([runner1, runner2, runner3], "classification")
    soft.train(
        train_ds,
        epochs=1,
        batch_size=32,
        checkpoints=[
            "../../data/checkpoints/runner1/", "../../data/checkpoints/runner2/", "../../data/checkpoints/runner3/"
        ]
    )

    img, label = next(iter(DataLoader(test_ds, batch_size=32)))
    pred = soft.predict(img)
    print("predict: ", pred)
    print("label:   ", label.numpy())

    total = 0.0
    correct = 0.0
    for data, label in DataLoader(test_ds, batch_size=32, shuffle=False):
        total += label.shape[0]

        predicts = soft.predict(data)
        correct += np.sum(predicts == label.numpy()).item()

    print("ens", correct / total)
    print("r1", soft.runners[0].evaluate(test_ds, y=None, batch_size=32))
    print("r2", soft.runners[1].evaluate(test_ds, y=None, batch_size=32))
    print("r3", soft.runners[2].evaluate(test_ds, y=None, batch_size=32))


if __name__ == '__main__':
    main()
