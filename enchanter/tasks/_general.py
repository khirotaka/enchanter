import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.optim as optim
import pytorch_lightning as pl


__all__ = ["_GeneralModule"]


class _GeneralModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: _Loss
    ):
        super(_GeneralModule, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        return loss

    def configure_optimizers(self):
        return self.optimizer
