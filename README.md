# Enchanter

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/84197fb283924f02a1667cea49dd031a)](https://app.codacy.com/manual/KawashimaHirotaka/enchanter?utm_source=github.com&utm_medium=referral&utm_content=khirotaka/enchanter&utm_campaign=Badge_Grade_Dashboard)
![CI testing](https://github.com/khirotaka/enchanter/workflows/CI/badge.svg)
![license](https://img.shields.io/github/license/khirotaka/enchanter?color=light)
![code size](https://img.shields.io/github/languages/code-size/khirotaka/enchanter?color=light)
[![Using PyTorch](https://img.shields.io/badge/PyTorch-red.svg?labelColor=f3f4f7&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iMjU2cHgiIGhlaWdodD0iMzEwcHgiIHZpZXdCb3g9IjAgMCAyNTYgMzEwIiB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHByZXNlcnZlQXNwZWN0UmF0aW89InhNaWRZTWlkIj4KICAgIDxnPgogICAgICAgIDxwYXRoIGQ9Ik0yMTguMjgxMDM3LDkwLjEwNjQxMiBDMjY4LjU3Mjk4OCwxNDAuMzk4MzYzIDI2OC41NzI5ODgsMjIxLjA3NTAzNCAyMTguMjgxMDM3LDI3MS43MTYyMzUgQzE2OS4wMzY4MzUsMzIyLjAwODE4NiA4OC4wMTA5MTQxLDMyMi4wMDgxODYgMzcuNzE4OTYzMiwyNzEuNzE2MjM1IEMtMTIuNTcyOTg3NywyMjEuNDI0Mjg0IC0xMi41NzI5ODc3LDE0MC4zOTgzNjMgMzcuNzE4OTYzMiw5MC4xMDY0MTIgTDEyNy44MjUzNzUsMCBMMTI3LjgyNTM3NSw0NS4wNTMyMDYgTDExOS40NDMzODMsNTMuNDM1MTk3OCBMNTkuNzIxNjkxNywxMTMuMTU2ODg5IEMyMi4wMDI3Mjg1LDE1MC4xNzczNTMgMjIuMDAyNzI4NSwyMTAuOTQ2Nzk0IDU5LjcyMTY5MTcsMjQ4LjY2NTc1NyBDOTYuNzQyMTU1NSwyODYuMzg0NzIgMTU3LjUxMTU5NiwyODYuMzg0NzIgMTk1LjIzMDU1OSwyNDguNjY1NzU3IEMyMzIuOTQ5NTIzLDIxMS42NDUyOTMgMjMyLjk0OTUyMywxNTAuODc1ODUzIDE5NS4yMzA1NTksMTEzLjE1Njg4OSBMMjE4LjI4MTAzNyw5MC4xMDY0MTIgWiBNMTczLjIyNzgzMSw4NC41MTg0MTc1IEMxNjMuOTY5MzM4LDg0LjUxODQxNzUgMTU2LjQ2Mzg0Nyw3Ny4wMTI5MjYzIDE1Ni40NjM4NDcsNjcuNzU0NDMzOCBDMTU2LjQ2Mzg0Nyw1OC40OTU5NDEzIDE2My45NjkzMzgsNTAuOTkwNDUwMiAxNzMuMjI3ODMxLDUwLjk5MDQ1MDIgQzE4Mi40ODYzMjMsNTAuOTkwNDUwMiAxODkuOTkxODE0LDU4LjQ5NTk0MTMgMTg5Ljk5MTgxNCw2Ny43NTQ0MzM4IEMxODkuOTkxODE0LDc3LjAxMjkyNjMgMTgyLjQ4NjMyMyw4NC41MTg0MTc1IDE3My4yMjc4MzEsODQuNTE4NDE3NSBaIiBmaWxsPSIjRUU0QzJDIj48L3BhdGg+CiAgICA8L2c+Cjwvc3ZnPgo=)](https://pytorch.org/)

Machine Learning Pipeline, Training and Logging for Me.

## Installation

```shell script
pip install git+https://github.com/khirotaka/enchanter.git
```

## Documentation
*   [Main Page](https://khirotaka.github.io/enchanter/)
      *   [addons](https://khirotaka.github.io/enchanter/docs/api/addons)
      *   [callbacks](https://khirotaka.github.io/enchanter/docs/api/callbacks)
      *   [engine](https://khirotaka.github.io/enchanter/docs/api/engine)
      *   [metrics](https://khirotaka.github.io/enchanter/docs/api/metrics)
      *   [wrappers](https://khirotaka.github.io/enchanter/docs/api/wrappers)

## Example

### Runner

#### run()
```python
from comet_ml import Experiment
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import enchanter.addons as addons
import enchanter.wrappers as wrappers


class MNIST(nn.Module):
    """
    MNIST 用のCNN
    """
    def __init__(self):
        super(MNIST, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            addons.Swish(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            addons.Swish(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*5*5, 512),
            addons.Swish(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, 64*5*5)
        out = self.fc(out)
        return out



experiment = Experiment()

train_ds = MNIST("./data", train=True, transform=transforms.ToTensor())
test_ds = MNIST("./data", train=False, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

model = MNIST()
optimizer = optim.Adam(model.parameters())
runner = wrappers.ClassificationRunner(
    model,
    optimizer,
    nn.CrossEntropyLoss(),
    experiment,
    scheduler=optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
)
runner.add_loader(train_loader, "train").add_loader(test_loader, "test")

runner.train_config(epochs=1)
runner.run(verbose=True)

```

### Comet.ml hyper parameter turning

```python
from comet_ml import Optimizer
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
import enchanter.wrappers as wrappers
import enchanter.addons as addons
import enchanter.addons.layers as layers
from enchanter.utils import comet

config = comet.TunerConfigGenerator(
    algorithm="bayes",
    metric="train_avg_loss",
    objective="minimize",
    seed=0,
    trials=5
)

config.suggest_categorical("activation", ["addons.mish", "torch.relu", "torch.sigmoid"])

opt = Optimizer(config.generate())

for experiment in opt.get_experiments():
    model = layers.MLP([4, 512, 128, 3], eval(experiment.get_parameter("activation")))
    optimizer = optim.Adam(model.parameters())
    runner = wrappers.ClassificationRunner(
        model, optimizer=optimizer, criterion=nn.CrossEntropyLoss(), experiment=experiment
    )
    x, y = load_iris(return_X_y=True)
    x = x.astype("float32")
    y = y.astype("int64")

    runner.fit(x, y, epochs=1)
```
