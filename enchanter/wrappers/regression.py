# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

import torch
from sklearn.metrics import r2_score
from sklearn.base import RegressorMixin

import enchanter
import enchanter.engine.modules as modules


__all__ = [
    "RegressionRunner"
]


class RegressionRunner(enchanter.engine.BaseRunner, RegressorMixin):
    """
    回帰問題を対象にしたRunner。

    Examples:
        >>> runner = RegressionRunner(...)
        >>> runner.add_loader("train", ...)
        >>> runner.train_config(epochs=1)
        >>> runner.run()
        >>> # OR
        >>> runner = RegressionRunner(...)
        >>> runner.fit(x, y, epochs=1, batch_size=32)
    """
    def __init__(self, model, optimizer, criterion, experiment, scheduler=None, early_stop=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.experiment = experiment
        self.scheduler = scheduler
        self.early_stop = early_stop

    def train_step(self, batch):
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)
        r2 = r2_score(y.cpu().numpy(), out.cpu().detach().numpy())
        return {"loss": loss, "r2": r2}

    def train_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_r2 = torch.stack([torch.tensor(x["r2"]) for x in outputs]).mean()
        return {"avg_loss": avg_loss, "avg_r2": avg_r2}

    def val_step(self, batch):
        return self.train_step(batch)

    def val_end(self, outputs):
        return self.train_end(outputs)

    def test_step(self, batch):
        return self.train_step(batch)

    def test_end(self, outputs):
        return self.train_end(outputs)

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = modules.numpy2tensor(x).to(self.device)
            out = self.model(x)

        return out.cpu().numpy()
