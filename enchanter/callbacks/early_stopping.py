# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

from typing import Dict
import numpy as np
from .base import Callback


__all__ = [
    "EarlyStopping"
]


class EarlyStopping(Callback):
    def __init__(self, monitor: str = "validate_avg_loss", min_delta=0.0, patience: int = 0, mode: str = "auto"):
        Callback.__init__(self)
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0

        mode_dict = {
            "min": np.less,
            "max": np.greater,
            "auto": np.greater if "acc" in self.monitor else np.less
        }
        if mode not in mode_dict:
            mode = "auto"

        self.monitor_op = mode_dict[mode]
        self.min_delta *= 1 if self.monitor_op == np.greater else -1
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def check_metrics(self, logs):
        monitor_val = logs.get(self.monitor)

        if monitor_val is None:
            return False

        return True

    def on_epoch_end(self, metrics: Dict, epoch: int) -> bool:
        stop = False
        if not self.check_metrics(metrics):
            return stop

        current = metrics.get(self.monitor)
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                stop = True

        return stop
