from abc import ABC, ABCMeta
from typing import Any, Dict, Tuple, Union, List, Optional

from numpy import ndarray
from tqdm import tqdm as tqdm
from tqdm.notebook import tqdm_notebook as tqdm_notebook
from sklearn.base import BaseEstimator
from torch.nn import Module
from torch.tensor import Tensor
from torch._C.device import device as torch_device
from torch.utils.data import DataLoader as DataLoader
from torch.optim.optimizer import Optimizer as Optimizer
from torch.optim.lr_scheduler import _LRScheduler as _LRScheduler
from comet_ml.experiment import BaseExperiment as BaseExperiment

from enchanter.callbacks import BaseLogger as BaseLogger
from enchanter.callbacks import EarlyStopping as EarlyStopping


class BaseRunner(BaseEstimator, ABC, metaclass=ABCMeta):
    model: Module = ...
    optimizer: Optimizer = ...
    experiment: Union[BaseExperiment, BaseLogger] = ...
    device: torch_device = ...
    pbar: Optional[Union[tqdm, tqdm_notebook]] = ...
    scheduler: Optional[_LRScheduler] = ...
    early_stop: Optional[EarlyStopping] = ...
    configures: Dict[str, Any] = ...

    _loaders: Dict[str, DataLoader] = ...
    _metrics: Dict = ...

    def __init__(self) -> None: ...
    def backward(self, loss: Tensor) -> None: ...
    def update_optimizer(self) -> None: ...
    def train_step(self, batch: Tuple) -> Dict[str, Tensor]: ...
    def train_end(self, outputs: List) -> Dict[str, Tensor]: ...
    def val_step(self, batch: Tuple) -> Dict[str, Tensor]: ...
    def val_end(self, outputs: List) -> Dict[str, Tensor]: ...
    def test_step(self, batch: Tuple) -> Dict[str, Tensor]: ...
    def test_end(self, outputs: List) -> Dict[str, Tensor]: ...
    def train_cycle(self, loader: DataLoader) -> None: ...
    def val_cycle(self, loader: DataLoader) -> None: ...
    def test_cycle(self, loader: DataLoader) -> None: ...
    def train_config(self, epochs: int, checkpoint_path: Optional[str] = ..., monitor: Optional[str] = ...) -> None: ...
    def log_hyperparams(self, dic: Dict=..., prefix: str=...) -> None: ...
    def initialize(self) -> None: ...
    def run(self, phase: str = ..., verbose: bool = ...) -> None: ...
    def predict(self, x: Union[Tensor, ndarray]) -> ndarray: ...
    def add_loader(self, mode: str, loader: DataLoader): ...
    @property
    def loaders(self) -> Dict[str, DataLoader]: ...
    def fit(self, x: ndarray, y: ndarray, **kwargs) -> None: ...
    def freeze(self) -> None: ...
    def unfreeze(self) -> None: ...
