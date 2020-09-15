# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

from typing import Any
from numpy import greater, less, Inf
from .base import Callback


__all__ = ["EarlyStopping"]


class EarlyStopping(Callback):
    def __init__(
        self,
        monitor: str = "validate_avg_loss",
        min_delta=0.0,
        patience: int = 0,
        mode: str = "auto",
    ) -> None:
        Callback.__init__(self)
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

    def check_metrics(self, logs: Any):
        monitor_val = logs.get(self.monitor)

        if monitor_val is None:
            return False

        return True

    def on_epoch_end(self, epoch, logs=None) -> bool:
        stop = False

        if logs:
            if not self.check_metrics(logs):
                return stop

            current = logs.get(self.monitor)
            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    stop = True

        return stop
