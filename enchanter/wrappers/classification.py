# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************


import torch
from sklearn.base import ClassifierMixin

import enchanter
import enchanter.engine.modules as modules
from enchanter.metrics import calculate_accuracy as calculate_accuracy


__all__ = [
    "ClassificationRunner"
]


class ClassificationRunner(enchanter.engine.BaseRunner, ClassifierMixin):
    """
    分類タスク向けの Runner です。

    Examples:
        >>> from comet_ml import Experiment
        >>> model = torch.nn.Sequential(...)
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> criterion = torch.nn.CrossEntropyLoss()
        >>> runner = ClassificationRunner(
        >>>     model,
        >>>     optimizer,
        >>>     criterion,
        >>>     Experiment()
        >>> )
        >>> runner.fit(...)
        >>> # or
        >>> runner.add_loader(...)
        >>> runner.train_config(epochs=10)
        >>> runner.run()

    """
    def __init__(self, model, optimizer, criterion, experiment, scheduler=None, early_stop=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.experiment = experiment
        self.criterion = criterion
        self.scheduler = scheduler
        self.early_stop = early_stop

    def train_step(self, batch):
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)
        accuracy = calculate_accuracy(out, y)
        return {"loss": loss, "accuracy": accuracy}

    def train_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([torch.tensor(x["accuracy"]) for x in outputs]).mean()
        return {"avg_loss": avg_loss, "avg_acc": avg_acc}

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
            _, predicted = torch.max(out, 1)

        return predicted.cpu().numpy()
