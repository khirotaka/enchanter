from typing import Tuple, List, Union, Optional

import torch
import numpy as np
from sklearn.base import ClassifierMixin
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from comet_ml.experiment import BaseExperiment as BaseExperiment

import enchanter
from enchanter.callbacks import EarlyStopping
from enchanter.callbacks import BaseLogger


class ClassificationRunner(enchanter.engine.BaseRunner, ClassifierMixin):
    model: torch.nn.Module = ...
    optimizer: Optimizer = ...
    criterion: _Loss = ...
    experiment: Union[BaseExperiment, BaseLogger] = ...

    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: Optimizer,
            criterion: _Loss,
            experiment: Union[BaseExperiment, BaseLogger],
            scheduler: Optional[_LRScheduler] = None,
            early_stop: Optional[EarlyStopping] = None
    ) -> None:
        super().__init__()
        ...

    def train_step(self, batch: Tuple): ...
    def train_end(self, outputs: List): ...
    def val_step(self, batch: Tuple): ...
    def val_end(self, outputs: List): ...
    def test_step(self, batch: Tuple): ...
    def test_end(self, outputs: List): ...
    def predict(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray: ...
