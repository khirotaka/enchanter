# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

from abc import ABC
from typing import Any


__all__ = ["Callback", "Dummy"]


class Callback(ABC):
    def __init__(self):
        self.stop_runner: bool = False

    def on_epoch_start(self, runner):
        """Called when the epoch begins."""
        pass

    def on_epoch_end(self, runner) -> Any:
        """Called when the epoch ends."""
        pass

    def on_step_start(self, runner) -> Any:
        """Called when the training batch begins."""
        pass

    def on_step_end(self, runner) -> Any:
        """Called when the training batch ends."""
        pass

    def on_train_start(self, runner) -> Any:
        """Called when the train begins."""
        pass

    def on_train_end(self, runner) -> Any:
        """Called when the train ends."""
        pass

    def on_validation_start(self, runner) -> Any:
        """Called when the validation loop begins."""
        pass

    def on_validation_end(self, runner) -> Any:
        """Called when the validation loop ends."""
        pass

    def on_test_start(self, runner) -> Any:
        """Called when the test begins."""
        pass

    def on_test_end(self, runner) -> Any:
        """Called when the test ends."""
        pass


class Dummy(Callback):
    pass
