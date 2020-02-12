# Enchanter

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/84197fb283924f02a1667cea49dd031a)](https://app.codacy.com/manual/KawashimaHirotaka/enchanter?utm_source=github.com&utm_medium=referral&utm_content=khirotaka/enchanter&utm_campaign=Badge_Grade_Dashboard)

Machine Learning Pipeline, Training and Logging for Me.

## System Requirements
* Python 3.7 or later
* PyTorch v1.4 or later


## Example

### Runner
Enchanter Runners has `train()` method and sklearn style `fit()` method.

#### train()
```python
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST

import enchanter
import enchanter.addon as addon


model = nn.Sequential(
    nn.Linear(4, 32),
    addon.Swish(),
    nn.Linear(32, 10)
)

ds = MNIST("./data", train=True)
runner = enchanter.ClassificationRunner(model, nn.CrossEntropyLoss(), optim.Adam, {"lr": 0.001})
runner.train(ds, epochs=10, batch_size=64)
```

#### fit()
```python
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris

import enchanter
import enchanter.addon as addon


model = nn.Sequential(
    nn.Linear(4, 32),
    addon.Swish(),
    nn.Linear(32, 10)
)

x, y = load_iris(return_X_y=True)
runner = enchanter.ClassificationRunner(model, nn.CrossEntropyLoss(), optim.Adam, {"lr": 0.001})
runner.fit(x, y)
```
