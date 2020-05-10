import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import enchanter
from enchanter.callbacks import TensorBoardLogger

X = torch.randn(512, 10).numpy()
Y = torch.randint(0, 9, (512,)).numpy()


class Runner1(enchanter.engine.BaseRunner):
    def __init__(self):
        super(Runner1, self).__init__()
        self.model = nn.Linear(10, 10)
        self.optimizer = optim.Adam(self.model.parameters())
        self.experiment = TensorBoardLogger("../tmp")
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, batch):
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)

        return {"loss": loss}


class Runner2(enchanter.engine.BaseRunner):
    def __init__(self):
        super(Runner2, self).__init__()
        self.model = nn.Linear(10, 10)
        self.optimizer = optim.Adam(self.model.parameters())
        self.experiment = TensorBoardLogger("../tmp")
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, batch):
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)

        return {"loss": loss}

    def val_step(self, batch):
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)

        return {"loss": loss}

    def test_step(self, batch):
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)

        return {"loss": loss}


def test_custom_runner_1():
    runner = Runner1()
    ds = enchanter.engine.modules.get_dataset(X, Y)
    loader = DataLoader(ds, batch_size=32)

    try:
        runner.add_loader("train", loader)
        runner.run()
        is_pass = True

    except Exception as e:
        print(e)
        is_pass = False

    assert is_pass


def test_custom_runner_2():
    runner = Runner2()
    ds = enchanter.engine.modules.get_dataset(X, Y)
    loader = DataLoader(ds, batch_size=32)

    try:
        runner.add_loader("test", loader)
        runner.run()
        is_pass = True

    except Exception as e:
        print(e)
        is_pass = False

    assert is_pass


def test_custom_runner_3():
    runner = Runner2()

    ds = enchanter.engine.modules.get_dataset(X, Y)
    loader = DataLoader(ds, batch_size=32)

    try:
        runner.add_loader("train", loader)
        runner.add_loader("val", loader)
        runner.add_loader("test", loader)
        runner.run()
        is_pass = True

    except Exception as e:
        print(e)
        is_pass = False

    assert is_pass
