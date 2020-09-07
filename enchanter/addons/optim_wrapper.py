# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

from typing import Dict, Any
import numpy as np
from torch.optim.optimizer import Optimizer


__all__ = ["TransformerOptimizer"]


class TransformerOptimizer:
    """
    Reference:
        `jadore801120/attention-is-all-you-need-pytorch \
        <https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py>`_

    """

    def __init__(self, optimizer: Optimizer, d_model: int, warm_up: int) -> None:
        self._optimizer: Optimizer = optimizer
        self.warm_up: int = warm_up
        self.n_current_steps: int = 0
        self.init_lr: float = np.power(d_model, -0.5)

    def step(self) -> None:
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self) -> None:
        self._optimizer.zero_grad()

    def _get_lr_scale(self) -> np.ndarray:
        return np.min(
            [
                np.power(self.n_current_steps, -0.5),
                np.power(self.warm_up, -1.5) * self.n_current_steps,
            ]
        )

    def get_lr(self) -> float:
        lr = self.init_lr * self._get_lr_scale()
        return lr

    def _update_learning_rate(self) -> None:
        self.n_current_steps += 1
        lr = self.get_lr()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

    def state_dict(self) -> Dict[str, Any]:
        return self._optimizer.state_dict()
