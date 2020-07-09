from typing import Union, Optional, OrderedDict, Dict

from comet_ml.experiment import BaseExperiment
from torch.nn.modules import Module
from torch.optim.optimizer import Optimizer

from enchanter.callbacks.loggers import BaseLogger


class RunnerIO:
    model: Module
    optimizer: Optimizer
    experiment: Union[BaseExperiment, BaseLogger]
    _checkpoint_path: Optional[str]

    def save_checkpoint(self) -> Dict[str, OrderedDict]: ...
    def load_checkpoint(self, checkpoint: Dict[str, OrderedDict]) -> None: ...
    def save(self, directory: Optional[str] = ..., epoch: Optional[int] = ...) -> None: ...
    def load(self, filename: str, map_location: str = ...) -> None: ...
