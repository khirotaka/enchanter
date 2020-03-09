import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import enchanter.wrappers as wrappers
import enchanter.addons as addons
import enchanter.engine as engine
from enchanter.callbacks import TensorBoardLogger

x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=0)

train_ds = engine.get_dataset(x_train.astype(np.float32), y_train.astype(np.int64))
val_ds = engine.get_dataset(x_val.astype(np.float32), y_val.astype(np.int64))
test_ds = engine.get_dataset(x_test.astype(np.float32), y_test.astype(np.int64))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=True)

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
    runner.add_loader("train", train_loader).add_loader("val", val_loader).add_loader("test", test_loader)
    runner.train_config(epochs=1)

    try:
        runner.run(verbose=True)
        is_pass = True
    except Exception:
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
