import torch
import enchanter
import enchanter.engine.modules as modules
from enchanter.metrics import accuracy as accuracy_func


class ClassificationRunner(enchanter.BaseRunner):
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

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = modules.numpy2tensor(x).float().to(self.device)
            out = self.model(x)
            _, predicted = torch.max(out, 1)

        return predicted.cpu().numpy()