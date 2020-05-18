import abc
from collections import OrderedDict
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union, List, Optional

import torch
import numpy as np
from tqdm import tqdm as tqdm
from sklearn import base as base

from torch.utils.data import DataLoader as DataLoader
from torch.optim.optimizer import Optimizer as Optimizer
from tqdm.notebook import tqdm_notebook as tqdm_notebook
from comet_ml.experiment import BaseExperiment as BaseExperiment
from torch.optim.lr_scheduler import _LRScheduler as _LRScheduler

from enchanter.engine import modules as modules
from enchanter.callbacks import BaseLogger as BaseLogger
from enchanter.callbacks import EarlyStopping as EarlyStopping


class BaseRunner(base.BaseEstimator, ABC, metaclass=abc.ABCMeta):
    model: torch.nn.Module = ...
    optimizer: Optimizer = ...
    experiment: Union[BaseExperiment, BaseLogger] = ...
    device: torch.device = ...
    pbar: Optional[Union[tqdm, tqdm_notebook]] = ...
    scheduler: Optional[_LRScheduler] = ...
    early_stop: Optional[EarlyStopping] = ...

    _epochs: int = ...
    _loaders: Dict[str, DataLoader] = ...
    _metrics: Dict = ...
    _checkpoint_path: str = ...

    def __init__(self) -> None: ...
    @abstractmethod
    def train_step(self, batch: Tuple) -> Dict[str, torch.Tensor]: ...
    def train_end(self, outputs: List) -> Dict[str, torch.Tensor]: ...
    def val_step(self, batch: Tuple) -> Dict[str, torch.Tensor]: ...
    def val_end(self, outputs: List) -> Dict[str, torch.Tensor]: ...
    def test_step(self, batch: Tuple) -> Dict[str, torch.Tensor]: ...
    def test_end(self, outputs: List) -> Dict[str, torch.Tensor]: ...
    def train_cycle(self, epoch: int, loader: DataLoader) -> None: ...
    def val_cycle(self, epoch: int, loader: DataLoader) -> None: ...
    def test_cycle(self, loader: DataLoader) -> None: ...
    def train_config(self, epochs: int, **kwargs: Any) -> None: ...
    def log_hyperparams(self, dic: Dict=..., prefix: str=...) -> None: ...
    def initialize(self) -> None: ...
    def run(self, verbose: bool = ...) -> None: ...
    def predict(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray: ...
    def add_loader(self, mode: str, loader: torch.utils.data.DataLoader): ...
    @property
    def loaders(self) -> Dict[str, DataLoader]: ...
    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None: ...
    def freeze(self) -> None: ...
    def unfreeze(self) -> None: ...
    def save_checkpoint(self) -> Dict[str, OrderedDict]: ...
    def load_checkpoint(self, checkpoint: Dict[str, OrderedDict]) -> None: ...
    def save(self, directory: Optional[str] = ..., epoch: Optional[int] = ...) -> None: ...
    def load(self, filename: str, map_location: str = ...) -> None: ...

