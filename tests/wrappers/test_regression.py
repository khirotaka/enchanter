import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

import enchanter.wrappers as wrappers
import enchanter.addons.layers as layers
from enchanter.addons import Mish
from enchanter.callbacks import TensorBoardLogger
from enchanter.engine.modules import get_dataset


x, y = load_boston(return_X_y=True)
y = y.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=0)

train_ds = get_dataset(x_train.astype(np.float32), y_train.astype(np.float32))
val_ds = get_dataset(x_val.astype(np.float32), y_val.astype(np.float32))
test_ds = get_dataset(x_test.astype(np.float32), y_test.astype(np.float32))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

model = layers.MLP([13, 512, 128, 1], Mish())
optimizer = optim.Adam(model.parameters())


def test_regression_1():
    runner = wrappers.RegressionRunner(
        model,
        optimizer,
        nn.MSELoss(),
        TensorBoardLogger("./logs"),
    )
    runner.add_loader("train", train_loader).add_loader("val", val_loader).add_loader("test", test_loader)
    runner.train_config(epochs=1)

    try:
        runner.run(verbose=True)
        is_pass = True
    except Exception as e:
        print(e)
        is_pass = False

    assert is_pass is True


def test_regression_2():
    runner = wrappers.RegressionRunner(
        model,
        optimizer,
        nn.MSELoss(),
        TensorBoardLogger("./logs")
    )
    runner.train_config(epochs=1)

    try:
        runner.run(verbose=False)
        is_pass = True

    except Exception:
        is_pass = False

    assert is_pass is False


def test_regression_3():
    runner = wrappers.RegressionRunner(
        model,
        optimizer,
        nn.MSELoss(),
        TensorBoardLogger("./logs")
    )
    try:
        runner.fit(x.astype(np.float32), y.astype(np.float32), batch_size=32, epochs=1)
        is_pass = True

    except Exception:
        is_pass = False

    assert is_pass is True
