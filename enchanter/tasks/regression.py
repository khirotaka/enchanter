# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

from typing import Tuple, List, Union, Optional, Dict

import numpy as np
from sklearn.metrics import r2_score
import torch
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.cuda.amp import GradScaler, autocast

from enchanter.engine import BaseRunner
from enchanter.callbacks import Callback


__all__ = ["RegressionRunner", "AutoEncoderRunner"]


class RegressionRunner(BaseRunner):
    """
    Runner for regression problems.

    Examples:
        >>> runner = RegressionRunner(...)
        >>> runner.add_loader("train", ...)
        >>> runner.train_config(epochs=1)
        >>> runner.run()
        >>> # OR
        >>> runner = RegressionRunner(...)
        >>> runner.fit(x, y, epochs=1, batch_size=32)
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion: _Loss,
        experiment,
        scheduler: Optional[List] = None,
        callbacks: Optional[List[Callback]] = None,
    ) -> None:
        super(RegressionRunner, self).__init__()
        self.model: Module = model
        self.optimizer: Optimizer = optimizer
        self.criterion: _Loss = criterion
        self.experiment = experiment
        if scheduler is None:
            self.scheduler: List = list()
        else:
            self.scheduler = scheduler
        self.callbacks = callbacks

    def general_step(self, batch: Tuple) -> Dict:
        """
        This method is executed by train_step, val_step, test_step.

        Args:
            batch:

        Returns:

        """
        x, y = batch

        with autocast(enabled=isinstance(self.scaler, GradScaler)):
            out = self.model(x)
            loss = self.criterion(out, y)

        r2 = r2_score(y.cpu().numpy(), out.cpu().detach().numpy())
        return {"loss": loss, "r2": r2}

    @staticmethod
    def general_end(outputs: List) -> Dict:
        """
        This method is executed by train_end, val_end, test_end.

        Args:
            outputs:

        Returns:

        """
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_r2 = torch.stack([torch.tensor(x["r2"]) for x in outputs]).mean()
        return {"avg_loss": avg_loss, "avg_r2": avg_r2}

    def train_step(self, batch: Tuple) -> Dict:
        return self.general_step(batch)

    def train_end(self, outputs: List) -> Dict:
        return self.general_end(outputs)

    def val_step(self, batch: Tuple) -> Dict:
        return self.general_step(batch)

    def val_end(self, outputs: List) -> Dict:
        return self.general_end(outputs)

    def test_step(self, batch: Tuple) -> Dict:
        return self.general_step(batch)

    def test_end(self, outputs: List) -> Dict:
        return self.general_end(outputs)

    def predict(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x = torch.as_tensor(x, device=self.device)
            out = self.model(x)

        return out.cpu().numpy()


class AutoEncoderRunner(RegressionRunner):
    """
    Runner for training AutoEncoder.

    """

    def general_step(self, batch: Tuple) -> Dict:
        """
        This method is executed by train_step, val_step, test_step.

        Args:
            batch:

        Returns:

        """
        x, _ = batch
        with autocast(enabled=isinstance(self.scaler, GradScaler)):
            out = self.model(x)
            loss = self.criterion(out, x)

        return {"loss": loss}

    @staticmethod
    def general_end(outputs: List) -> Dict:
        """
        This method is executed by train_end, val_end, test_end.

        Args:
            outputs:

        Returns:

        """
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        return {"avg_loss": avg_loss}
