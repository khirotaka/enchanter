# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

from typing import Any, Optional, Dict, Union, Iterator, List
from abc import ABC, abstractmethod
from contextlib import contextmanager

import numpy as np
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


__all__ = ["BaseLogger", "TensorBoardLogger"]


class BaseLogger(ABC):
    """
    Provides minimal compatibility with the `comet_ml.Experiment`, which is required to run Runner.

    """

    def __init__(self) -> None:
        self.context: Union[str, None] = None

    def add_tag(self, tag: str):
        pass

    def add_tags(self, tags: List[str]):
        pass

    @contextmanager
    def train(self) -> Iterator:
        old_state = self.context
        self.context = "train"

        yield self

        self.context = old_state

    @contextmanager
    def validate(self) -> Iterator:
        old_state = self.context
        self.context = "validate"

        yield self

        self.context = old_state

    @contextmanager
    def test(self) -> Iterator:
        old_state = self.context
        self.context = "test"

        yield self

        self.context = old_state

    @abstractmethod
    def log_metric(
        self,
        name: str,
        value: Any,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        include_context: bool = True,
    ) -> None:
        """
        Logs a metric.

        Args:
            name: name of metric
            value:
            step:
            epoch:
            include_context:

        Returns:
            None

        Warnings:
            If you create your own Logger, you will need to implement this method.

        """
        raise NotImplementedError

    @abstractmethod
    def log_metrics(
        self,
        dic: Dict,
        prefix: Optional[str] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> None:
        """

        Logs a key, value dictionary of metrics.

        See Also:
            log_metric


        Warnings:
            If you create your own Logger, you will need to implement this method.

        """
        raise NotImplementedError

    @abstractmethod
    def log_parameter(self, name: str, value: Any, step: Optional[int] = None) -> None:
        """
        Logs a single hyper-parameter.

        Args:
            name: name of hyper-parameter
            value: value
            step:

        Warnings:
            If you create your own Logger, you will need to implement this method.

        """
        raise NotImplementedError

    @abstractmethod
    def log_parameters(self, dic: Dict, prefix: Optional[str] = None, step: Optional[int] = None) -> None:
        """
        Logs a key, value dictionary of hyper-parameters.

        See Also:
            log_peramter


        Warnings:
            If you create your own Logger, you will need to implement this method.

        """
        raise NotImplementedError

    def set_model_graph(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def end(self):
        """

        Use to indicate that the experiment is complete.

        Warnings:
            If you create your own Logger, you will need to implement this method.

        """
        raise NotImplementedError

    def log_model(self, name, file_or_folder, file_name=None, overwrite=False, metadata=None, copy_to_tmp=True):
        pass

    def log_asset(self, file_data, file_name=None, overwrite=False, copy_to_tmp=True, step=None, metadata=None):
        pass

    def log_asset_data(self, data, name=None, overwrite=False, step=None, metadata=None, file_name=None):
        pass

    def log_asset_folder(self, folder, step=None, log_file_name=False, recursive=False):
        pass


def _value2str(value) -> str:
    if isinstance(value, Tensor):
        value = value.cpu().item()
    elif isinstance(value, np.ndarray):
        value = value.item()

    return str(value)


class TensorBoardLogger(BaseLogger):
    """
    TenorBoardLogger is a module that supports the minimum logging in the environment where comet.ml cannot be used.

    Examples:
        >>> from enchanter.tasks import ClassificationRunner
        >>> model, optimizer, criterion = ...
        >>> runner = ClassificationRunner(
        >>>     model, optimizer, criterion, experiment=TensorBoardLogger()
        >>> )

    """

    def __init__(self, *args, **kwargs):
        """
        See the initializer argument for `torch.utils.tensorboard.writer.SummaryWriter
        <https://pytorch.org/docs/master/tensorboard.html>`_ .
        """
        super(TensorBoardLogger, self).__init__()
        self.writer: SummaryWriter = SummaryWriter(*args, **kwargs)

    def log_metric(
        self,
        name: str,
        value: Any,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        include_context: bool = True,
    ) -> None:
        """
        Logs a metric.

        Args:
            name: name of metric
            value: value
            step: Used as the X axis when plotting on TensorBoard.
            epoch: Used as the X axis when plotting on TensorBoard.
            include_context:

        Returns:
            None

        """
        if step is not None:
            tmp = step
        else:
            if epoch is not None:
                tmp = epoch
            else:
                tmp = 0

        if step is not None and epoch is None:
            tmp = step
        elif step is None and epoch is not None:
            tmp = epoch

        if isinstance(value, list):
            value = np.array(value)

        self.writer.add_scalar("{}/{}".format(name, self.context), value, global_step=tmp)

    def log_metrics(
        self,
        dic: Dict,
        prefix: Optional[str] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> None:
        """
        Logs a key, value dictionary of metrics.

        See Also:
            log_metric

        """
        for k, v in dic.items():
            if prefix:
                k = "{}_{}".format(prefix, k)
            self.log_metric(name=k, value=v, step=step, epoch=epoch)

    def log_parameter(self, name: Optional[str], value: Any, step: Optional[int] = None) -> None:
        """
        Logs a single hyper-parameter.

        Args:
            name: name of hyper-parameter
            value: value
            step: used as the X-axis when plotting on TensorBoard.

        Returns:

        """
        value = _value2str(value)
        if self.context:
            prefix = self.context
        else:
            prefix = "Hyperparameters"

        if name:
            table = "|key|value|\n|-|-|\n|{}|{}|  \n".format(name, value)
        else:
            table = value

        self.writer.add_text("{}/{}".format(prefix, name), table, global_step=step)

    def log_parameters(self, dic: Dict, prefix: Optional[str] = None, step: Optional[int] = None) -> Any:
        """
        Logs a key, value dictionary of hyper-parameters.

        See Also:
            log_peramter

        """
        table = "|key|value|\n|-|-|\n"
        for k, v in dic.items():
            table += "|{}|{}|  \n".format(k, v)

        self.log_parameter(None, table, step)

    def end(self):
        """
        Use to indicate that the experiment is complete.

        """
        self.writer.close()
