# Enchanter
Machine Learning Pipeline, Training and Logging for Me.

## System Requirements
* Python 3.7 or later
* PyTorch v1.3 or later


## Example

### Runner

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from enchanter import estimator

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


train_ds: Dataset = ...

runner = estimator.ClassificationRunner(Model, nn.CrossEntropyLoss(), torch.optim.Adam, {"lr": 0.001})
runner.fit(train_ds, epochs=10, batch_size=32, shuffle=True)

```

### Ensemble

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from enchanter.estimator.ensemble import SoftEnsemble, HardEnsemble
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

def create_ds():
    train_ds = MNIST("../data", train=True, download=False, transform=ToTensor())
    test_ds = MNIST("../data", train=False, download=False, transform=ToTensor())
    return train_ds, test_ds

def create_models():
    runner1 = ClassificationRunner(Model(), nn.CrossEntropyLoss(), optim.Adam, optim_conf={"lr": 0.001})
    runner2 = ClassificationRunner(Model(), nn.CrossEntropyLoss(), optim.Adam, optim_conf={"lr": 0.002})
    runner3 = ClassificationRunner(Model(), nn.CrossEntropyLoss(), optim.Adam, optim_conf={"lr": 0.003})
    
    return runner1, runner2, runner3

def soft():
    train_ds, test_ds = create_ds()
    runner1, runner2, runner3 = create_models()

    ensemble = SoftEnsemble([runner1, runner2, runner3], "classification")
    ensemble.fit(train_ds, epochs=1, batch_size=32)

    img, label = next(iter(DataLoader(test_ds, batch_size=32)))
    print("predict: ", ensemble.predict(img).astype("int"))
    print("label", label)

def hard():
    train_ds, test_ds = create_ds()
    runner1, runner2, runner3 = create_models()

    ensemble = HardEnsemble([runner1, runner2, runner3])
    ensemble.fit(train_ds, epochs=1, batch_size=32)

    img, label = next(iter(DataLoader(test_ds, batch_size=32)))
    print("predict: ", ensemble.predict(img).astype("int"))
    print("label", label)


if __name__ == '__main__':
    soft()
    hard()

```