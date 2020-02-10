import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

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

    hard = ensemble.HardEnsemble([runner1, runner2, runner3])
    hard.train(train_ds, epochs=1, batch_size=32)

    img, label = next(iter(DataLoader(test_ds, batch_size=32)))
    print("predict: ", hard.predict(img).astype("int"))
    print("label", label)


if __name__ == '__main__':
    main()
