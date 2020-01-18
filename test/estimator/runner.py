import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

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
    train_ds = MNIST("../data", train=True, download=True, transform=ToTensor())
    test_ds = MNIST("../data", train=False, download=True, transform=ToTensor())

    model = Model()
    runner = ClassificationRunner(model, nn.CrossEntropyLoss(), optim.Adam, optim_conf={"lr": 0.001})
    runner.fit(train_ds, epochs=2, batch_size=64, checkpoint="../data/checkpoint/")
    loss, accuracy = runner.evaluate(test_ds)

    img, label = next(iter(DataLoader(test_ds, batch_size=1)))
    print(runner.predict(img))
    print(label)
    print("Loss: {:.4f}".format(loss))
    print("Accuracy: {:.4%}".format(accuracy))

    print("Save")
    runner.save("../data/checkpoint/")

    print("load")
    runner.load("../data/checkpoint/checkpoint_epoch_1.pth")


if __name__ == '__main__':
    main()
