# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

import io
import os
import re
import operator
from abc import ABC
from time import sleep
from collections import OrderedDict
from typing import Any, Dict, Tuple, Union, List, Optional

from tqdm.auto import tqdm
from numpy import floor, ndarray

import torch
from torch.nn import Module
from torch import Tensor
from torch.cuda import is_available, amp
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, SubsetRandomSampler

try:
    from comet_ml import Experiment
    from comet_ml.api import APIExperiment

    _COMET_AVAILABLE = True

except ImportError:
    _COMET_AVAILABLE = False

from enchanter.utils.backend import is_scalar
from enchanter.callbacks import Callback
from enchanter.callbacks import CallbackManager
from enchanter.engine.saving import RunnerIO
from enchanter.engine.modules import send, get_dataset, is_tfds, tfds_to_numpy, restore_state_dict


__all__ = ["BaseRunner"]


class BaseRunner(ABC, RunnerIO):
    """
    A class for creating runners to train PyTorch models.


    Examples:

        >>> from comet_ml import Experiment
        >>> import torch
        >>> class Runner(BaseRunner):
        >>>     def __init__(self):
        >>>         super(Runner, self).__init__()
        >>>         self.model = torch.nn.Linear(10, 10)
        >>>         self.optimizer = torch.optim.Adam(self.model.parameters())
        >>>         self.experiment = Experiment()
        >>>         self.criterion = torch.nn.CrossEntropyLoss()
        >>>
        >>>     def train_step(self, batch):
        >>>         x, y = batch
        >>>         out = self.model(x)
        >>>         loss = self.criterion(out, y)
        >>>
        >>>         return {"loss": loss}

    """

    def __init__(self) -> None:
        super(BaseRunner, self).__init__()
        self.device: torch.device = torch.device("cuda" if is_available() else "cpu")
        self.model: Module = NotImplemented
        self.optimizer: Optimizer = NotImplemented
        self.scheduler: List = list()
        self.experiment = NotImplemented
        self.manager = CallbackManager()
        self.callbacks: Optional[List[Callback]] = None
        self.scaler: Optional[amp.GradScaler] = None
        self.api_experiment: Optional = None

        self.global_step: int = 0
        self.non_blocking: bool = True
        self.configures: Dict[str, Any] = {"epochs": 0, "checkpoint_path": None}
        self.pbar: Union[tqdm, range] = range(self.configures["epochs"])
        self.metrics: Dict = {"train": dict(), "val": dict(), "test": dict()}
        self._loaders: Dict[str, Union[DataLoader, Any]] = dict()

    def backward(self, loss: Tensor) -> None:
        """
        calculate the gradient.
        If self.scaler is a `torch.cuda.amp.GradScaler` object, it is automatically processed by amp.

        Args:
            loss (torch.Tensor):

        Returns:
            None

        """
        if isinstance(self.scaler, amp.GradScaler):
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def update_optimizer(self) -> None:
        """
        update optimizer

        """
        if isinstance(self.scaler, amp.GradScaler):
            self.scaler.step(self.optimizer)
            self.scaler.update()

        else:
            self.optimizer.step()

    def update_scheduler(self, epoch: int) -> None:
        """
        Method called to update the value of the scheduler.
        It is called after updating the Optimizer.

        Args:
            epoch (int): Current Epoch

        Returns:
            None

        Examples:
            >>> from torch.optim import lr_scheduler
            >>> from enchanter.tasks import ClassificationRunner
            >>> runner: BaseRunner = ClassificationRunner(
            >>>     model=...,
            >>>     optimizer=...,
            >>>     criterion=...,
            >>>     scheduler=[
            >>>         lr_scheduler.CosineAnnealingLR(...), lr_scheduler.StepLR(...)
            >>>     ]
            >>> )

        """
        for scheduler in self.scheduler:
            scheduler.step()

        self.experiment.log_metric("scheduler_lr", self.scheduler[-1].get_last_lr(), epoch=epoch)

    def train_step(self, batch: Tuple) -> Dict[str, Tensor]:
        """
        When training the neural network,
            >>> import torch.nn as nn
            >>> train_loader: DataLoader = ...
            >>> model: nn.Module = ...
            >>> criterion = ...
            >>> for x, y in train_loader:
            >>>     out = model(x)
            >>>     loss = criterion(out, y)

        this method is responsible for the above areas.

        Args:
            batch: A tuple containing data & labels get from the PyTorch DataLoader.

        Returns:
            You need to return a dictionary with the key 'loss'.

        Examples:
            >>> def train_step(self, batch):
            >>>     x, y = batch
            >>>     out = self.model(x)
            >>>     loss = nn.functional.cross_entropy(out, y)
            >>>     return {"loss": loss}

        """

    def train_end(self, outputs: List) -> Dict[str, Tensor]:
        """
        This method is executed at the end of each step of neural network training.

        Args:
            outputs:

        Returns:

        """
        return {}

    def val_step(self, batch: Tuple) -> Dict[str, Tensor]:
        """
        This method is executed at every 1 step when validating the neural net.
        See `train_step()` for help.


        Args:
            batch: A tuple containing data & labels get from the PyTorch DataLoader.

        Returns:
            You need to return a dictionary.

        """

    def val_end(self, outputs: List) -> Dict[str, Tensor]:
        """
        This method is executed at the end of each step of neural network validating.

        Args:
            outputs:

        Returns:
            You need to return a dictionary.

        """
        return {}

    def test_step(self, batch: Tuple) -> Dict[str, Tensor]:
        """
        This method is executed at every 1 step when testing the neural net.
        See `train_step()` for help.

        Args:
            batch: A tuple containing data & labels get from the PyTorch DataLoader.

        Returns:
            You need to return a dictionary.

        """

    def test_end(self, outputs: List) -> Dict[str, Tensor]:
        """
        This method is executed at the end of each step of neural network testing.

        Args:
            outputs:

        Returns:
            You need to return a dictionary.

        """
        return {}

    def train_cycle(self, epoch: int, loader: DataLoader) -> None:
        """
        This is a training loop for neural net.

        Args:
            epoch
            loader (torch.utils.data.DataLoader):

        """
        results = list()
        loader_size = len(loader)

        if is_tfds(loader):
            loader = tfds_to_numpy(loader)

        self.model.train()
        with self.experiment.train():
            for step, batch in enumerate(loader):
                self.optimizer.zero_grad()
                batch = send(batch, self.device, self.non_blocking)
                # on_step_start()
                self.global_step += 1
                outputs = self.train_step(batch)
                self.backward(outputs["loss"])
                self.update_optimizer()

                if hasattr(self.pbar, "set_postfix"):
                    per = "{:1.0%}".format(step / loader_size)
                    self.pbar.set_postfix(OrderedDict(train_batch=per), refresh=True)  # type: ignore

                outputs = {
                    key: outputs[key].detach().cpu() if isinstance(outputs[key], Tensor) else outputs[key]
                    for key in outputs.keys()
                }
                tmp = {k: outputs[k] for k in outputs.keys() if is_scalar(outputs[k])}
                self.experiment.log_metrics(tmp, step=self.global_step, epoch=epoch)
                results.append(outputs)
                self.manager.on_train_step_end(outputs)

            dic = self.train_end(results)  # pylint: disable=E1111

            if len(dic) != 0:
                self.metrics["train"].update(dic)
                self.experiment.log_metrics(dic, step=epoch, epoch=epoch)

    def val_cycle(self, epoch: int, loader: DataLoader) -> None:
        """
        This is a validating loop for neural net.

        Args:
            epoch
            loader:

        Returns:

        """
        results = list()
        loader_size = len(loader)

        if is_tfds(loader):
            loader = tfds_to_numpy(loader)

        self.model.eval()
        with self.experiment.validate(), torch.no_grad():
            for step, batch in enumerate(loader):
                batch = send(batch, self.device, self.non_blocking)
                self.global_step += 1
                # on_step_start()
                outputs = self.val_step(batch)  # pylint: disable=E1111

                if hasattr(self.pbar, "set_postfix"):
                    per = "{:1.0%}".format(step / loader_size)
                    self.pbar.set_postfix(OrderedDict(val_batch=per), refresh=True)  # type: ignore

                outputs = {
                    key: outputs[key].cpu() if isinstance(outputs[key], Tensor) else outputs[key]
                    for key in outputs.keys()
                }
                tmp = {k: outputs[k] for k in outputs.keys() if is_scalar(outputs[k])}
                self.experiment.log_metrics(tmp, step=self.global_step, epoch=epoch)
                results.append(outputs)
                self.manager.on_validation_step_end(outputs)

            dic = self.val_end(results)  # pylint: disable=E1111

            if len(dic) != 0:
                self.metrics["val"].update(dic)
                self.experiment.log_metrics(dic, step=epoch, epoch=epoch)

    def test_cycle(self, loader: DataLoader) -> None:
        """
        This is a testing loop for neural net.

        Args:
            loader:

        Returns:

        """
        results = list()
        loader_size = len(loader)

        if is_tfds(loader):
            loader = tfds_to_numpy(loader)

        self.model.eval()
        with self.experiment.test(), torch.no_grad():
            for step, batch in enumerate(loader):
                batch = send(batch, self.device, self.non_blocking)
                # on_step_start()
                outputs = self.test_step(batch)  # pylint: disable=E1111

                per = "{:1.0%}".format(step / loader_size)
                if hasattr(self.pbar, "set_postfix"):
                    self.pbar.set_postfix(OrderedDict(test_batch=per), refresh=True)  # type: ignore

                    self.pbar.update(1)  # type: ignore

                outputs = {
                    key: outputs[key].cpu() if isinstance(outputs[key], Tensor) else outputs[key]
                    for key in outputs.keys()
                }

                tmp = {k: outputs[k] for k in outputs.keys() if is_scalar(outputs[k])}
                self.experiment.log_metrics(tmp)
                results.append(outputs)
                self.manager.on_test_step_end(outputs)

            dic = self.test_end(results)  # pylint: disable=E1111

            if len(dic) != 0:
                self.metrics["test"].update(dic)
                self.experiment.log_metrics(dic)

    def train_config(
        self,
        epochs: int,
        checkpoint_path: Optional[str] = None,
        monitor: Optional[str] = None,
    ):
        """
        This method is used to specify epochs and so on when you execute using the .run() method.

        Examples:
            >>> runner: BaseRunner = ...
            >>> runner.train_config(
            >>>     epochs=10,
            >>>     checkpoint_path="/path/to/checkpoint_dir",
            >>>     monitor="validate_avg_acc >= 0.75"
            >>> )

        Args:
            epochs (int): Specify the number of training epochs.
            checkpoint_path: Specify the name of the directory where the checkpoint is stored,
                             and if monitor is not specified, store weights for all epochs.
            monitor: Save only the epoch that corresponds to the specified expression.
                     The `checkpoint_path` must be set together with it.

        Returns:
            None
        Notes:
            When you specify the monitor argument, be sure to put a space between 'keyword', 'symbol', and 'value'.

        """

        self.configures["checkpoint_path"] = checkpoint_path
        if monitor:
            try:
                _ = re.search("train|validate", monitor)[0]  # type: ignore
            except TypeError:
                raise KeyError("The argument monitor is not an expected expression. {}".format(monitor))
            else:
                self.configures["monitor"] = monitor

            if _COMET_AVAILABLE and not isinstance(self.experiment, Experiment):
                raise TypeError(
                    "To use `.train_config(, monitor='....')`, you need an `Experiment` object. `experiment` is {}.\
                    ".format(
                        type(self.experiment)
                    )
                )

        if epochs > 0:
            self.configures["epochs"] = epochs
        else:
            self.configures["epochs"] = 1

        return self

    def log_hyperparams(self, dic: Dict = None, prefix: Optional[str] = None) -> None:
        """
        logging hyper parameters

        Args:
            dic (Dict):
            prefix (str):

        Returns:
            None

        """
        self.experiment.log_parameters(self.optimizer.__dict__["defaults"], prefix="optimizer")
        self.experiment.log_parameter("Optimizer", self.optimizer.__class__.__name__)

        if dic is not None:
            self.experiment.log_parameters(dic, prefix)

    def initialize(self) -> None:
        """
        The method that prepares the Runner.
        If the variables required for execution, such as self.model, self.optimizer, self.experiment, etc.,
        are not defined, the program will exit with an error message.
        If there are no problems, pass the model to the CPU or GPU.

        Returns:
            None

        """

        if not isinstance(self.model, Module):
            raise NotImplementedError("`self.model` is not defined.")

        if not isinstance(self.optimizer, Optimizer):
            raise NotImplementedError("`self.optimizer` is not defined.")

        if self.experiment is NotImplemented:
            raise NotImplementedError("`self.experiment` is not defined.")

        if self.scheduler and not isinstance(self.scheduler, list):
            raise ValueError("`scheduler` must be a list object.")

        if _COMET_AVAILABLE and isinstance(self.experiment, Experiment):
            self.api_experiment = APIExperiment(previous_experiment=self.experiment.id, cache=False)

        if self.global_step < 0:
            self.global_step = 0

        self.manager.callbacks = self.callbacks

        self.manager.set_experiment(self.experiment)
        self.manager.set_device(self.device)
        self.manager.set_optimizer(self.optimizer)
        self.manager.set_model(self.model)

        self.model = self.model.to(self.device)
        self.save_dir = self.configures["checkpoint_path"]

    def run(self, phase: str = "all", verbose: bool = True, sleep_time: int = 1):
        """
        Runners are executed.
        To run it, you must register a data loader using self.add_loader() before.

        Args:
            phase (str):
                - `train`
                - `val`
                - `test`
                - `all`
                - `debug`

                by specifying one of the above, you can determine the execution phase. Default: all

            verbose (bool): If true, progress is displayed.
            sleep_time (int): The time to wait for data transfer to the comet.ml server (in seconds). Default: 1 (sec).

        Notes:
            If "TypeError" occurs even though there is no mistake in the formula of the monitor specified by
            `.train_config()`, the reason may be that it takes a long time to transfer the data to the comet.ml server.
            Try to set the `sleep_time` to about 5 seconds.

        Returns:
            None

        """
        phases = {"train", "train/val", "test", "all", "debug"}
        if phase not in phases:
            raise KeyError("The argument 'phase' must be one of the following. {}".format(phases))

        if phase == "debug":
            if hasattr(self.experiment, "add_tag"):
                self.experiment.add_tag("debug")  # type: ignore

        self.initialize()
        self.log_hyperparams()

        if not self.loaders:
            raise ValueError("At least one DataLoader must be provided.")

        if phase in {"all", "train", "train/val", "debug"}:
            if "train" in self.loaders:
                self.pbar = (
                    tqdm(range(self.configures["epochs"]), desc="Epochs")
                    if verbose
                    else range(self.configures["epochs"])
                )

                for epoch in self.pbar:
                    self.manager.on_epoch_start(epoch, self.metrics)

                    self.manager.on_train_start(self.metrics)
                    self.train_cycle(epoch, self.loaders["train"])
                    self.manager.on_train_end(self.metrics)

                    if phase in {"all", "train/val", "debug"}:
                        if "val" in self.loaders:
                            self.manager.on_validation_start(self.metrics)
                            self.val_cycle(epoch, self.loaders["val"])
                            self.manager.on_validation_end(self.metrics)

                    if self.scheduler:
                        self.update_scheduler(epoch)

                    self.manager.on_epoch_end(epoch, self.metrics)

                    if self.manager.stop_runner:
                        if self.manager.params["best_weight"]:
                            self.model = restore_state_dict(self.model, self.manager.params["best_weight"])
                        if hasattr(self.pbar, "close"):
                            self.pbar.close()  # type: ignore
                        break

                    if self.configures["checkpoint_path"]:
                        ops = {
                            "==": operator.eq,
                            "!=": operator.ne,
                            "<": operator.lt,
                            "<=": operator.le,
                            ">": operator.gt,
                            ">=": operator.ge,
                        }

                        if "monitor" in self.configures.keys():
                            key, op, value = self.configures["monitor"].split(" ")
                            value = float(value)
                            sleep(sleep_time)

                            try:
                                current_value = float(
                                    self.api_experiment.get_metrics_summary(key)["valueCurrent"]  # type: ignore
                                )
                            except TypeError:
                                raise KeyError(
                                    "The specified key was not found. Check the settings of `.train_config()`."
                                )

                            if current_value:
                                if ops[op](current_value, value):
                                    self.save(self.configures["checkpoint_path"], epoch=epoch)

                        else:
                            self.save(self.configures["checkpoint_path"], epoch=epoch)

        if phase in {"all", "test", "debug"}:
            if "test" in self.loaders:
                self.manager.on_test_start(self.metrics)
                self.pbar = tqdm(total=len(self.loaders["test"]), desc="Evaluating") if verbose else None
                self.test_cycle(self.loaders["test"])
                self.manager.on_test_end(self.metrics)

        return self

    def predict(self, x: Union[Tensor, ndarray]) -> ndarray:
        """
        A method that makes predictions based on the given input.

        Args:
            x (Union[torch.Tensor, np.ndarray]):

        Returns:
            predict
        """
        raise NotImplementedError

    def add_loader(self, mode: str, loader: Union[DataLoader, Any]):
        """
        A method to register a DataLoader to be used for training etc. in a runner.

        Args:
            mode (str): Specify one of ['train', 'val', 'test'].
            loader (torch.utils.data.DataLoader):

        Examples:
            >>> train_loader = DataLoader(...)
            >>> runner: BaseRunner = ...
            >>> runner.add_loader("train", train_loader)

        """
        if mode not in ["train", "val", "test"]:
            raise KeyError("argument `mode` must be one of 'train', 'val', or 'test'.")

        if is_tfds(loader) or isinstance(loader, DataLoader):
            pass

        else:
            raise TypeError("The argument `loader` must be an instance of `torch.utils.data.DataLoader`.")

        self._loaders[mode] = loader
        return self

    @property
    def loaders(self) -> Dict[str, DataLoader]:
        if len(self._loaders) != 0:
            return self._loaders
        else:
            raise ValueError

    def fit(self, x: ndarray, y: ndarray, **kwargs):
        """
        Scikit-Learn style training method.

        Args:
            x: Training data
            y: Label
            **kwargs:

        """
        val_size: float = kwargs.get("val_size", 0.1)
        num_workers: int = kwargs.get("num_workers", os.cpu_count())
        batch_size: int = kwargs.get("batch_size", 1)
        pin_memory: bool = kwargs.get("pin_memory", False)
        verbose: bool = kwargs.get("verbose", True)
        checkpoint_path = kwargs.get("checkpoint_path", None)
        monitor = kwargs.get("monitor", None)

        if self.configures["epochs"] == 0:
            epochs = kwargs.get("epochs", 1)
        else:
            epochs = self.configures["epochs"]

        train_ds = get_dataset(x, y)
        val_ds = get_dataset(x, y)
        n_train = len(train_ds)
        indices = list(range(n_train))
        split = int(floor(val_size * n_train))

        train_idx, val_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.add_loader("train", train_loader)
        self.add_loader("val", val_loader)
        self.train_config(epochs, checkpoint_path=checkpoint_path, monitor=monitor)
        self.run(verbose=verbose)

        return self

    def freeze(self) -> None:
        """
        A method to freeze the model's parameters so that they do not calculate the slope.

        """
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

    def unfreeze(self) -> None:
        """
        The method to make it possible to re-learn the parameters fixed by `.freeze()`.

        """
        for param in self.model.parameters():
            param.requires_grad = True

        self.model.train()

    def quite(self) -> None:
        """
        Quit Runner.

        When this method is executed, it sends an exit command to `comet.ml`.

        """

        self.experiment.end()

    def __enter__(self):
        """
        Context API

        Examples:
            >>> runner: BaseRunner = ...
            >>> with runner:
            >>>     runner.add_loader(...)
            >>>     runner.train_config(...)
            >>>     runner.run()

        """
        self.initialize()
        self.log_hyperparams()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        At the end of the ``with`` block, the Experiment will end automatically.

        """
        buffer = io.BytesIO()
        torch.save(self.save_checkpoint(), buffer)
        self.experiment.log_asset_data(
            buffer.getvalue(),
            step=self.global_step,
            name="context_api/enchanter_checkpoints_latest.pth",
        )
        self.quite()
