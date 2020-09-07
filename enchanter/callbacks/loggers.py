# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

from typing import Any, Optional, Dict, Union, Iterator
from contextlib import contextmanager
from numpy import ndarray
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


__all__ = ["BaseLogger", "TensorBoardLogger"]


class BaseLogger:
    def __init__(self) -> None:
        self.context: Union[str, None] = None

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

    def log_metric(
        self,
        name: str,
        value: Any,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        include_context: bool = True,
    ) -> None:
        pass

    def log_metrics(
        self,
        dic: Dict,
        prefix: Optional[str] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> None:
        pass

    def log_parameter(self, name: str, value: Any, step: Optional[int] = None) -> None:
        pass

    def log_parameters(self, dic: Dict, prefix: Optional[str] = None, step: Optional[int] = None) -> None:
        pass

    def set_model_graph(self, *args: Any, **kwargs: Any) -> None:
        pass


class TensorBoardLogger(BaseLogger):
    def __init__(self, log_dir, *args, **kwargs):
        super(TensorBoardLogger, self).__init__()
        self.writer: SummaryWriter = SummaryWriter(log_dir, *args, **kwargs)

    def log_metric(
        self,
        name: str,
        value: Any,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        include_context: bool = True,
    ) -> None:
        self.writer.add_scalar("{}/{}".format(self.context, name), value)

    def log_metrics(
        self,
        dic: Dict,
        prefix: Optional[str] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> None:
        for k, v in dic.items():
            self.writer.add_scalar(k, v)

    def log_parameter(self, name: str, value: Any, step: Optional[int] = None) -> None:
        if (
            isinstance(value, Tensor)
            or isinstance(value, ndarray)
            or isinstance(value, float)
            or isinstance(value, int)
        ):
            self.writer.add_scalar("{}/{}/{}".format(self.context, "params", name), value, step)

    def log_parameters(self, dic: Dict, prefix: Optional[str] = None, step: Optional[int] = None) -> Any:
        for k, v in dic.items():
            self.log_parameter(k, v, step)
