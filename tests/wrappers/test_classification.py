import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from torch.utils.data import DataLoader

import enchanter.wrappers as wrappers
import enchanter.addons as addons
import enchanter.engine as engine
from enchanter.callbacks import TensorBoardLogger

x, y = load_iris(return_X_y=True)
ds = engine.get_dataset(x.astype(np.float32), y.astype(np.int64))
loader = DataLoader(ds, batch_size=32, shuffle=True)

model = nn.Sequential(
    nn.Linear(4, 16),
    addons.Mish(),
    nn.Linear(16, 3)
)
optimizer = optim.Adam(model.parameters())


def test_classification_1():
    runner = wrappers.ClassificationRunner(
        model,
        optimizer,
        nn.CrossEntropyLoss(),
        TensorBoardLogger("./logs")
    )
    runner.add_loader("train", loader)
    runner.train_config(epochs=1)

    try:
        runner.run(verbose=True)
        is_pass = True
    except Exception as e:
        is_pass = False

    assert is_pass is True


def test_classification_2():
    runner = wrappers.ClassificationRunner(
        model,
        optimizer,
        nn.CrossEntropyLoss(),
        TensorBoardLogger("./logs")
    )
    runner.train_config(epochs=1)

    try:
        runner.run(verbose=False)
        is_pass = True

    except Exception:
        is_pass = False

    assert is_pass is False
