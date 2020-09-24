from typing import List, Optional, Dict
from .base import Callback, Dummy


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
        if callbacks is not None:
            self.callbacks = callbacks
        else:
            self.callbacks = [Dummy()]

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)

    def set_optimizer(self, optimizer):
        for callback in self.callbacks:
            callback.set_optimizer(optimizer)

    def flag_check(self, stop_runner: bool):
        if stop_runner:
            self.stop_runner = stop_runner

    def on_epoch_start(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_start(epoch, logs)

            self.flag_check(callback.stop_runner)

    def on_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

            self.flag_check(callback.stop_runner)

    def on_train_step_start(self, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_train_step_start(logs)

            self.flag_check(callback.stop_runner)

    def on_train_step_end(self, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_train_step_end(logs)

            self.flag_check(callback.stop_runner)

    def on_validation_step_start(self, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_validation_step_start(logs)

            self.flag_check(callback.stop_runner)

    def on_validation_step_end(self, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_validation_step_end(logs)

            self.flag_check(callback.stop_runner)

    def on_test_step_start(self, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_test_step_start(logs)

            self.flag_check(callback.stop_runner)

    def on_test_step_end(self, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_test_step_end(logs)

            self.flag_check(callback.stop_runner)

    def on_train_start(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_start(logs)

            self.flag_check(callback.stop_runner)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

            self.flag_check(callback.stop_runner)

    def on_validation_start(self, logs=None):
        for callback in self.callbacks:
            callback.on_validation_start(logs)

            self.flag_check(callback.stop_runner)

    def on_validation_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_validation_end(logs)

            self.flag_check(callback.stop_runner)

    def on_test_start(self, logs=None):
        for callback in self.callbacks:
            callback.on_test_start(logs)

            self.flag_check(callback.stop_runner)

    def on_test_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_test_end(logs)

            self.flag_check(callback.stop_runner)
