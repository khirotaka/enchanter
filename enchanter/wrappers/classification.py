import torch
import enchanter
from enchanter.metrics import accuracy as accuracy_func


class ClassificationRunner(enchanter.BaseRunner):
    def __init__(self, model, optimizer, criterion, experiment, scheduler=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.experiment = experiment
        self.criterion = criterion
        self.scheduler = scheduler

    def train_step(self, batch):
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)
        accuracy = accuracy_func(out, y)
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
