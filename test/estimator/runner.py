import comet_ml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import accuracy_score

import enchanter


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = MNIST("../data", train=True, download=True, transform=ToTensor())
    val_ds = MNIST("../data", train=True, download=True, transform=ToTensor())
    pin_memory = torch.cuda.is_available()

    val_size = 0.1
    n_trains = len(train_ds)
    indices = list(range(n_trains))
    splits = int(np.floor(val_size * n_trains))
    train_idx, val_idx = indices[splits:], indices[:splits]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    test_ds = MNIST("../data", train=False, download=True, transform=ToTensor())

    model = Model()
    runner = enchanter.ClassificationRunner(
        model, nn.CrossEntropyLoss(), optim.Adam, optim_config={"lr": 0.001}, 
        experiment=comet_ml.Experiment(project_name="testflight"),
        scheduler={
            "algorithm": optim.lr_scheduler.CosineAnnealingLR,
            "config": {"T_max": 50}
        }, device=device
    )

    runner.train(
        train_ds, epochs=2, batch_size=128, 
        checkpoint="../data/checkpoint/", 
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
    metrics = runner.evaluate(test_ds, batch_size=64, metrics=[accuracy_score])

    img, label = next(iter(DataLoader(test_ds, batch_size=32)))
    print(runner.predict(img))
    print(label)
    print("Loss: {:.4f}".format(metrics["loss"]))
    print("Accuracy: {:.4%}".format(metrics["accuracy_score"]))

    print("Save")
    runner.save("../data/checkpoint/")

    print("load")
    runner.load("../data/checkpoint/checkpoint_epoch_1.pth")


if __name__ == '__main__':
    main()
