# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

from typing import Any, Dict, Optional
from numpy import greater, less, Inf
from .base import Callback


__all__ = ["EarlyStopping"]


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
        self.monitor: Any = monitor
        self.patience: Any = patience
        self.min_delta: Any = min_delta
        self.wait: int = 0
        self.stopped_epoch: int = 0

        mode_dict = {
            "min": less,
            "max": greater,
            "auto": greater if "acc" in self.monitor else less,
        }
        if mode not in mode_dict:
            mode = "auto"

        self.monitor_op: Any = mode_dict[mode]
        self.min_delta *= 1 if self.monitor_op == greater else -1
        self.best: Any = Inf if self.monitor_op == less else -Inf

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
            cat_logs = {}
            for pk in logs.keys():
                for ck, cv in logs[pk].items():
                    cat_logs["{}_{}".format(pk, ck)] = cv

            if not self.check_metrics(cat_logs):
                self.stop_runner = stop
            else:
                current = cat_logs.get(self.monitor)
                if self.monitor_op(current - self.min_delta, self.best):
                    self.best = current
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        self.stopped_epoch = epoch
                        stop = True

                self.stop_runner = stop
