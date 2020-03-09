# Enchanter

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/84197fb283924f02a1667cea49dd031a)](https://app.codacy.com/manual/KawashimaHirotaka/enchanter?utm_source=github.com&utm_medium=referral&utm_content=khirotaka/enchanter&utm_campaign=Badge_Grade_Dashboard)
![CI testing](https://github.com/khirotaka/enchanter/workflows/CI/badge.svg)
![license](https://img.shields.io/github/license/khirotaka/enchanter?color=light)
![code size](https://img.shields.io/github/languages/code-size/khirotaka/enchanter?color=light)

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
