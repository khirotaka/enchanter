# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************
from typing import Tuple, List, Union, Optional, Dict

from numpy import ndarray
from sklearn.metrics import r2_score
from sklearn.base import RegressorMixin
from torch import Tensor
from torch.nn.modules import Module
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.cuda.amp import GradScaler, autocast
from torch import stack, tensor, no_grad, as_tensor
from comet_ml.experiment import BaseExperiment as BaseExperiment

from enchanter.engine import BaseRunner
from enchanter.callbacks import EarlyStopping, BaseLogger

__all__ = ["RegressionRunner"]


class RegressionRunner(BaseRunner, RegressorMixin):
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

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion: _Loss,
        experiment: Union[BaseExperiment, BaseLogger],
        scheduler: Optional[List] = None,
        early_stop: Optional[EarlyStopping] = None,
    ) -> None:
        super(RegressionRunner, self).__init__()
        self.model: Module = model
        self.optimizer: Optimizer = optimizer
        self.criterion: _Loss = criterion
        self.experiment: Union[BaseExperiment, BaseLogger] = experiment
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

        r2 = r2_score(y.cpu().numpy(), out.cpu().detach().numpy())
        return {"loss": loss, "r2": r2}

    @staticmethod
    def general_end(outputs: List) -> Dict:
        avg_loss = stack([x["loss"] for x in outputs]).mean()
        avg_r2 = stack([tensor(x["r2"]) for x in outputs]).mean()
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

    def predict(self, x: Union[Tensor, ndarray]) -> ndarray:
        self.model.eval()
        with no_grad():
            x = as_tensor(x, device=self.device)
            out = self.model(x)

        return out.cpu().numpy()


class AutoEncoderRunner(RegressionRunner):
    def general_step(self, batch: Tuple) -> Dict:
        x, _ = batch
        with autocast(enabled=isinstance(self.scaler, GradScaler)):
            out = self.model(x)
            loss = self.criterion(out, x)

        return {"loss": loss}

    @staticmethod
    def general_end(outputs: List) -> Dict:
        avg_loss = stack([x["loss"] for x in outputs]).mean()
        return {"avg_loss": avg_loss}
