from typing import List, Optional
from .base import Callback, Dummy


__all__ = ["CallbackManager"]


class CallbackManager(Callback):
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        super(CallbackManager, self).__init__()
        if callbacks is not None:
            self.callbacks = callbacks
        else:
            self.callbacks = [Dummy()]

    def flag_check(self, stop_runner: bool):
        if stop_runner:
            self.stop_runner = stop_runner

    def on_epoch_start(self, runner):
        for callback in self.callbacks:
            callback.on_epoch_start(runner)

            self.flag_check(callback.stop_runner)

    def on_epoch_end(self, runner):
        for callback in self.callbacks:
            callback.on_epoch_end(runner)

            self.flag_check(callback.stop_runner)

    def on_step_start(self, runner):
        for callback in self.callbacks:
            callback.on_step_start(runner)

            self.flag_check(callback.stop_runner)

    def on_step_end(self, runner):
        for callback in self.callbacks:
            callback.on_step_end(runner)

            self.flag_check(callback.stop_runner)

    def on_train_start(self, runner):
        for callback in self.callbacks:
            callback.on_train_start(runner)

            self.flag_check(callback.stop_runner)

    def on_train_end(self, runner):
        for callback in self.callbacks:
            callback.on_train_end(runner)

            self.flag_check(callback.stop_runner)

    def on_validation_start(self, runner):
        for callback in self.callbacks:
            callback.on_validation_start(runner)

            self.flag_check(callback.stop_runner)

    def on_validation_end(self, runner):
        for callback in self.callbacks:
            callback.on_validation_end(runner)

            self.flag_check(callback.stop_runner)

    def on_test_start(self, runner):
        for callback in self.callbacks:
            callback.on_test_start(runner)

            self.flag_check(callback.stop_runner)

    def on_test_end(self, runner):
        for callback in self.callbacks:
            callback.on_test_end(runner)

            self.flag_check(callback.stop_runner)
