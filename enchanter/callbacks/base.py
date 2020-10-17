# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

from abc import ABC
from typing import Dict, Optional


__all__ = ["Callback"]


class Callback(ABC):
    def __init__(self):
        self.stop_runner: bool = False
        self.model = None
        self.optimizer = None
        self.device = None
        self.experiment = None
        self.params = {}

    def set_device(self, device):
        self.device = device

    def set_model(self, model):
        self.model = model

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def on_epoch_start(self, epoch, logs: Optional[Dict] = None):
        """Called when the epoch begins."""
        pass

    def on_epoch_end(self, epoch, logs: Optional[Dict] = None):
        """Called when the epoch ends."""
        pass

    def on_train_step_start(self, logs: Optional[Dict] = None):
        """Called when the training batch begins."""
        pass

    def on_train_step_end(self, logs: Optional[Dict] = None):
        """Called when the training batch ends. You can access the output of train_step."""
        pass

    def on_validation_step_start(self, logs: Optional[Dict] = None):
        pass

    def on_validation_step_end(self, logs: Optional[Dict] = None):
        """Called when the validation batch ends. You can access the output of val_step."""
        pass

    def on_test_step_start(self, logs: Optional[Dict] = None):
        pass

    def on_test_step_end(self, logs: Optional[Dict] = None):
        """Called when the test batch ends. You can access the output of test_step."""
        pass

    def on_train_start(self, logs: Optional[Dict] = None):
        """Called when the train begins."""
        pass

    def on_train_end(self, logs: Optional[Dict] = None):
        """Called when the train ends."""
        pass

    def on_validation_start(self, logs: Optional[Dict] = None):
        """Called when the validation loop begins."""
        pass

    def on_validation_end(self, logs: Optional[Dict] = None):
        """Called when the validation loop ends."""
        pass

    def on_test_start(self, logs: Optional[Dict] = None):
        """Called when the test begins."""
        pass

    def on_test_end(self, logs: Optional[Dict] = None):
        """Called when the test ends."""
        pass
