from typing import List, Optional, Dict
from .base import Callback


__all__ = ["CallbackManager"]


class CallbackManager(Callback):
    """
    A class for managing Callback.

    Examples:
        >>> from enchanter.callbacks import EarlyStopping
        >>> manager = CallbackManager([
        >>>     EarlyStopping()
        >>> ])
        >>> for epoch in range(5):
        >>>     logs = ...
        >>>     manager.on_epoch_end(epoch, logs)
        >>>     if manager.stop_runner:
        >>>         break

    """

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        super(CallbackManager, self).__init__()
        self.callbacks = callbacks

    def set_experiment(self, experiment):
        self.experiment = experiment
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.experiment = self.experiment

    def set_device(self, device):
        self.device = device
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.device = self.device

    def set_model(self, model):
        self.model = model
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.model = self.model

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.optimizer = self.optimizer

    def flag_check(self, stop_runner: bool):
        if stop_runner and self.callbacks is not None:
            self.stop_runner = stop_runner

    def on_epoch_start(self, epoch, logs=None):
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.on_epoch_start(epoch, logs)

                self.flag_check(callback.stop_runner)

    def on_epoch_end(self, epoch, logs=None):
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, logs)

                self.flag_check(callback.stop_runner)
                if "best_weight" in callback.params.keys():
                    self.params["best_weight"] = callback.params["best_weight"]

    def on_train_step_start(self, logs: Optional[Dict] = None):
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.on_train_step_start(logs)

                self.flag_check(callback.stop_runner)

    def on_train_step_end(self, logs: Optional[Dict] = None):
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.on_train_step_end(logs)

                self.flag_check(callback.stop_runner)

    def on_validation_step_start(self, logs: Optional[Dict] = None):
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.on_validation_step_start(logs)

                self.flag_check(callback.stop_runner)

    def on_validation_step_end(self, logs: Optional[Dict] = None):
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.on_validation_step_end(logs)

                self.flag_check(callback.stop_runner)

    def on_test_step_start(self, logs: Optional[Dict] = None):
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.on_test_step_start(logs)

                self.flag_check(callback.stop_runner)

    def on_test_step_end(self, logs: Optional[Dict] = None):
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.on_test_step_end(logs)

                self.flag_check(callback.stop_runner)

    def on_train_start(self, logs=None):
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.on_train_start(logs)

                self.flag_check(callback.stop_runner)

    def on_train_end(self, logs=None):
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.on_train_end(logs)

                self.flag_check(callback.stop_runner)

    def on_validation_start(self, logs=None):
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.on_validation_start(logs)

                self.flag_check(callback.stop_runner)

    def on_validation_end(self, logs=None):
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.on_validation_end(logs)

                self.flag_check(callback.stop_runner)

    def on_test_start(self, logs=None):
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.on_test_start(logs)

                self.flag_check(callback.stop_runner)
                if "grid_search" in callback.params.keys():
                    self.params["grid_search"] = callback.params["grid_search"]

    def on_test_end(self, logs=None):
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.on_test_end(logs)

                self.flag_check(callback.stop_runner)
