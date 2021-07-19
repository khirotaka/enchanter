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
import torch
from torch.nn.modules import Module
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.cuda.amp import GradScaler, autocast

from enchanter.engine import BaseRunner
from enchanter.callbacks import Callback
from enchanter.metrics import calculate_accuracy


__all__ = ["ClassificationRunner"]


class ClassificationRunner(BaseRunner):
    """
    Runner for classification tasks.

    Examples:
        >>> from comet_ml import Experiment
        >>> import torch
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

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion: _Loss,
        experiment,
        scheduler: Optional[List] = None,
        callbacks: Optional[List[Callback]] = None,
    ) -> None:
        super(ClassificationRunner, self).__init__()
        self.model: Module = model
        self.optimizer: Optimizer = optimizer
        self.experiment = experiment
        self.criterion: _Loss = criterion
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

        accuracy = calculate_accuracy(out, y)
        return {"loss": loss, "accuracy": accuracy}

    @staticmethod
    def general_end(outputs: List) -> Dict:
        """
        This method is executed by train_end, val_end, test_end.

        Args:
            outputs:

        Returns:

        """
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([torch.tensor(x["accuracy"]) for x in outputs]).mean()
        return {"avg_loss": avg_loss, "avg_acc": avg_acc}

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
            _, predicted = torch.max(out, 1)

        return predicted.cpu().numpy()
