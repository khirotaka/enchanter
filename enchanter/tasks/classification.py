# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

from typing import Tuple, List, Union, Optional, Dict

from sklearn.base import ClassifierMixin
from numpy import ndarray
from torch import Tensor
from torch.nn.modules import Module
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.cuda.amp import GradScaler, autocast
from torch import no_grad, stack, tensor, as_tensor, max as torch_max
from comet_ml.experiment import BaseExperiment as BaseExperiment

from enchanter.engine import BaseRunner
from enchanter.callbacks import BaseLogger
from enchanter.callbacks import EarlyStopping
from enchanter.metrics import calculate_accuracy as calculate_accuracy


__all__ = ["ClassificationRunner"]


class ClassificationRunner(BaseRunner, ClassifierMixin):
    """
    分類タスク向けの Runner です。

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
        experiment: Union[BaseExperiment, BaseLogger],
        scheduler: Optional[List] = None,
        early_stop: Optional[EarlyStopping] = None,
    ) -> None:
        super(ClassificationRunner, self).__init__()
        self.model: Module = model
        self.optimizer: Optimizer = optimizer
        self.experiment: Union[BaseExperiment, BaseLogger] = experiment
        self.criterion: _Loss = criterion
        if scheduler is None:
            self.scheduler: List = list()
        else:
            self.scheduler = scheduler

        self.early_stop = early_stop

    def general_step(self, batch: Tuple) -> Dict:
        x, y = batch

        with autocast(enabled=isinstance(self.scaler, GradScaler)):
            out = self.model(x)
            loss = self.criterion(out, y)

        accuracy = calculate_accuracy(out, y)
        return {"loss": loss, "accuracy": accuracy}

    @staticmethod
    def general_end(outputs: List) -> Dict:
        avg_loss = stack([x["loss"] for x in outputs]).mean()
        avg_acc = stack([tensor(x["accuracy"]) for x in outputs]).mean()
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

    def predict(self, x: Union[Tensor, ndarray]) -> ndarray:
        self.model.eval()
        with no_grad():
            x = as_tensor(x, device=self.device)
            out = self.model(x)
            _, predicted = torch_max(out, 1)

        return predicted.cpu().numpy()
