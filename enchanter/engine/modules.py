from typing import Dict
import torch
import numpy as np
from torch.utils.data import TensorDataset, Dataset


def is_jupyter() -> bool:
    if "get_ipython" not in globals():
        return False
    env = get_ipython().__class__.__name__
    if env == "TerminalInteractiveShell":
        return False
    return True


def numpy2tensor(inputs: np.ndarray) -> torch.Tensor:
    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs)

    return inputs


def get_dataset(x, y=None) -> Dataset:
    x = numpy2tensor(x)

    if y is not None:
        y = numpy2tensor(y)
        ds = TensorDataset(x, y)
    else:
        ds = TensorDataset(x)

    return ds


class CometLogger:
    def __init__(self, experiment):
        self.experiment = experiment

    def log_train(self, epoch: int, step: int, values: Dict):
        with self.experiment.train():
            for k in values.keys():
                self.experiment.log_metric(k, values[k], step=step, epoch=epoch)

    def log_val(self, epoch: int, step: int, values: Dict):
        with self.experiment.validate():
            for k in values.keys():
                self.experiment.log_metric(k, values[k], step=step, epoch=epoch)

    def log_test(self, values: Dict):
        with self.experiment.test():
            for k in values.keys():
                self.experiment.log_metric(k, values[k])
