import torch
import enchanter
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from typing import Any, Tuple, List, Union
from enchanter.callbacks import EarlyStopping

class ClassificationRunner(enchanter.BaseRunner):
    model: torch.nn.Module = ...
    optimizer: Optimizer = ...
    criterion: _Loss = ...
    experiment: Any = ...

    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: Optimizer,
            criterion: _Loss,
            experiment: Any,
            scheduler: Any,
            early_stop: Union[EarlyStopping, None]
    ) -> None:
        super().__init__()
        ...

    def train_step(self, batch: Tuple): ...
    def train_end(self, outputs: List): ...
    def val_step(self, batch: Tuple): ...
    def val_end(self, outputs: List): ...
    def test_step(self, batch: Tuple): ...
    def test_end(self, outputs: List): ...
    def predict(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray: