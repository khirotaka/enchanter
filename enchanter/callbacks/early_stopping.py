# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

from typing import Any, Dict, Optional, Union

import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, StratifiedKFold
from .base import Callback


__all__ = ["EarlyStopping", "EarlyStoppingForTSUS"]


class EarlyStopping(Callback):
    def __init__(
        self,
        monitor: str = "val_avg_loss",
        min_delta: float = 0.0,
        patience: int = 0,
        mode: str = "auto",
    ) -> None:
        """
        The training ends when the value to be monitored stops changing.

        Args:
            monitor: You can choose the return values of ``train_end()``, ``val_end()``, or ``test_end()``.
                     To specify, in the case of the return value of ``train_end()``, ``train_XXX``,
                     in the case of the return value of ``val_end()``, ``val_XXX``,
                     in the case of the return value of ``test_end()``, ``test_XXX``.

            min_delta: Minimum value of change determined as improvement for the monitored value.
            patience: If there is no improvement in the monitored value during the specified number of epochs,
                      the training stops.
            mode: One of {``'auto'``, ``'min'``, ``'max'``} is selected.

                - ``min`` mode ends the training when the decrease in the monitored value stops.
                - ``max`` mode, the training is terminated when the monitored values stop increasing.
                - ``auto`` mode, automatically estimated from the monitored values.

        """
        super(EarlyStopping, self).__init__()
        self.monitor: str = monitor
        self.patience: int = patience
        self.min_delta: float = min_delta
        self.wait: int = 0
        self.stopped_epoch: int = 0

        mode_dict = {
            "min": np.less,
            "max": np.greater,
            "auto": np.greater if "acc" in self.monitor else np.less,
        }
        if mode not in mode_dict:
            mode = "auto"

        self.monitor_op: Any = mode_dict[mode]
        self.min_delta *= 1 if self.monitor_op == np.greater else -1
        self.best: Any = np.Inf if self.monitor_op == np.less else -np.Inf

    def check_metrics(self, logs: Dict) -> bool:
        monitor_val = logs.get(self.monitor)

        if monitor_val is None:
            return False

        return True

    def on_epoch_end(self, epoch, logs: Optional[Dict] = None) -> None:
        stop = False

        if not isinstance(logs, dict):
            raise TypeError("The argument `logs` is not the expected data type.")
        else:
            cat_logs: Dict[str, Union[float, int]] = {}
            for pk in logs.keys():
                for ck, cv in logs[pk].items():
                    cat_logs["{}_{}".format(pk, ck)] = cv

            if not self.check_metrics(cat_logs):
                self.stop_runner = stop
            else:
                try:
                    current = cat_logs[self.monitor]
                except KeyError:
                    raise KeyError("Can't find the value specified in the argument `monitor`.")

                if self.monitor_op(current - self.min_delta, self.best):
                    self.best = current
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait > self.patience:
                        self.stopped_epoch = epoch
                        stop = True

                self.stop_runner = stop


class EarlyStoppingForTSUS(Callback):
    """
    Early Stopping for Time Series Unsupervised Runner
    """

    def __init__(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        classifier=SVC(),
        monitor: str = "accuracy",
        patience: int = 5,
        kfold=None,
    ):
        # TODO min_delta
        super(EarlyStoppingForTSUS, self).__init__()
        self.classifier = classifier
        self.data: torch.Tensor = data
        self.target: torch.Tensor = target
        self.patience = patience
        self.monitor: str = monitor
        self.best: float = 0.0
        self.wait: int = 0
        self.stopped_epoch: int = 0

        if kfold is not None:
            self.kfold = kfold
        else:
            self.kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    def encode(self):
        out = []
        self.model.eval()
        with torch.no_grad():
            self.data = self.data.to(self.device)
            out.append(self.model(self.data).cpu().numpy())

        return np.vstack(out)

    def cross_val(self):
        features: np.ndarray = self.encode()
        score = cross_validate(
            self.classifier, features, self.target.cpu().numpy(), cv=self.kfold, scoring=[self.monitor]
        )["test_{}".format(self.monitor)].mean()
        return score

    def on_epoch_end(self, epoch, logs: Optional[Dict] = None):
        stop = False
        current_score = self.cross_val()

        if current_score > self.best:
            self.best = current_score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait > self.patience:
                self.stopped_epoch = epoch
                stop = True

        self.stop_runner = stop
