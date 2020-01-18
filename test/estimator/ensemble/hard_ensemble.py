import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from enchanter.estimator.ensemble import HardEnsemble
from enchanter.estimator.runner import ClassificationRunner


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

    runner1 = ClassificationRunner(Model(), nn.CrossEntropyLoss(), optim.Adam, optim_conf={"lr": 0.001})
    runner2 = ClassificationRunner(Model(), nn.CrossEntropyLoss(), optim.Adam, optim_conf={"lr": 0.002})
    runner3 = ClassificationRunner(Model(), nn.CrossEntropyLoss(), optim.Adam, optim_conf={"lr": 0.003})

    ensemble = HardEnsemble([runner1, runner2, runner3])
    ensemble.fit(train_ds, epochs=1, batch_size=32)

    img, label = next(iter(DataLoader(test_ds, batch_size=32)))
    print("predict: ", ensemble.predict(img).astype("int"))
    print("label", label)


if __name__ == '__main__':
    main()
