# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

import re
import operator
from abc import ABC
from collections import OrderedDict
from typing import Any, Dict, Tuple, Union, List, Optional

from numpy import floor, ndarray
from tqdm.auto import tqdm
from torch import device
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.cuda import is_available
from torch.tensor import Tensor
from torch.autograd import no_grad
from torch.utils.data import DataLoader, SubsetRandomSampler
from comet_ml import Experiment, BaseExperiment, APIExperiment


from enchanter.engine.saving import RunnerIO
from enchanter.engine.modules import send, get_dataset
from enchanter.callbacks import BaseLogger as BaseLogger
from enchanter.callbacks import EarlyStopping as EarlyStopping


__all__ = [
    "BaseRunner"
]


class BaseRunner(ABC, RunnerIO):
    """
    PyTorchモデルの訓練に用いる Runner を作成する為のクラスです。


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
        self.device: device = device("cuda" if is_available() else "cpu")
        self.model: Module = NotImplemented
        self.optimizer: Optimizer = NotImplemented
        self.scheduler: Optional[List] = None
        self.experiment: Union[BaseExperiment, BaseLogger] = NotImplemented
        self.early_stop: Optional[EarlyStopping] = None
        self.configures: Dict[str, Any] = {
            "epochs": 0
        }
        self.api_experiment = None

        self.pbar = None
        self._loaders: Dict[str, DataLoader] = {}
        self._metrics: Dict = {}
        self.global_step: int = 0

    def backward(self, loss: Tensor) -> None:
        """
        calculate the gradient

        Warnings:
            This method may be changed from staticmethod in the future to support FP16 and others.

        Args:
            loss (torch.Tensor):

        Returns:
            None

        """
        loss.backward()

    def update_optimizer(self) -> None:
        self.optimizer.step()

    def update_scheduler(self, epoch: int) -> None:
        """
        スケジューラの値を更新するために呼ばれるメソッドです。
        Optimizerを更新した後に呼ばれます。

        Args:
            epoch (int): 現在のエポック数

        Returns:
            None

        Examples:
            >>> from torch.optim import lr_scheduler
            >>> from enchanter.wrappers import ClassificationRunner
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
        ニューラルネットの訓練時、
            >>> import torch.nn as nn
            >>> train_loader: DataLoader = ...
            >>> model: nn.Module = ...
            >>> criterion = ...
            >>> for x, y in train_loader:
            >>>     out = model(x)
            >>>     loss = criterion(out, y)

        にあたる箇所を担当するメソッドです。

        Args:
            batch: PyTorch DataLoader から得られる訓練用のデータトラベルを含むタプル

        Returns:
            キーに 'loss' を含む辞書を返す必要があります。

        Examples:
            >>> def train_step(self, batch):
            >>>     x, y = batch
            >>>     out = self.model(x)
            >>>     loss = nn.functional.cross_entropy(out, y)
            >>>     return {"loss": loss}

        """

    def train_end(self, outputs: List) -> Dict[str, Tensor]:
        """
        ニューラルネットの訓練時、1step 終了ごとに実行されるメソッドです。

        Args:
            outputs:

        Returns:

        """
        return {}

    def val_step(self, batch: Tuple) -> Dict[str, Tensor]:
        """
        ニューラルネットの検証時、1step ごとに実行されるメソッドです。

        Args:
            batch: PyTorch DataLoader から得られる訓練用のデータトラベルを含むタプル

        Returns:
            辞書を返す必要があります。

        """

    def val_end(self, outputs: List) -> Dict[str, Tensor]:
        """
        ニューラルネットの検証時、1step 終了ごとに実行されるメソッドです。

        Args:
            outputs:

        Returns:

        """
        return {}

    def test_step(self, batch: Tuple) -> Dict[str, Tensor]:
        """
        ニューラルネットの評価時、1step ごとに実行されるメソッドです。

        Args:
            batch: PyTorch DataLoader から得られる訓練用のデータトラベルを含むタプル

        Returns:
            辞書を返す必要があります。

        """

    def test_end(self, outputs: List) -> Dict[str, Tensor]:
        """
        ニューラルネットの評価時、1step 終了ごとに実行されるメソッドです。

        Args:
            outputs:

        Returns:

        """
        return {}

    def train_cycle(self, epoch: int, loader: DataLoader) -> None:
        """
        ニューラルネットの訓練ループです。

        Args:
            epoch
            loader (torch.utils.data.DataLoader):

        """
        results = list()
        loader_size = len(loader)

        self.model.train()
        with self.experiment.train():
            for step, batch in enumerate(loader):
                self.optimizer.zero_grad()
                batch = send(batch, self.device)
                # on_step_start()
                self.global_step += 1
                outputs = self.train_step(batch)
                self.backward(outputs["loss"])
                self.update_optimizer()

                if hasattr(self.pbar, "set_postfix"):
                    per = "{:1.0%}".format(step / loader_size)
                    self.pbar.set_postfix(
                        OrderedDict(train_batch=per), refresh=True
                    )

                outputs = {
                    key: outputs[key].detach().cpu() if isinstance(outputs[key], Tensor) else outputs[key]
                    for key in outputs.keys()
                }
                self.experiment.log_metrics(outputs, step=self.global_step, epoch=epoch)
                results.append(outputs)
                # on_step_end()

            dic = self.train_end(results)        # pylint: disable=E1111

            if len(dic) != 0:
                self._metrics.update(dic)
                self.experiment.log_metrics(dic, step=epoch, epoch=epoch)

    def val_cycle(self, epoch: int, loader: DataLoader) -> None:
        """
        ニューラルネットの評価用ループです。

        Args:
            epoch
            loader:

        Returns:

        """
        results = list()
        loader_size = len(loader)

        self.model.eval()
        with self.experiment.validate():
            with no_grad():
                for step, batch in enumerate(loader):
                    batch = send(batch, self.device)
                    self.global_step += 1
                    # on_step_start()
                    outputs = self.val_step(batch)        # pylint: disable=E1111

                    if hasattr(self.pbar, "set_postfix"):
                        per = "{:1.0%}".format(step / loader_size)
                        self.pbar.set_postfix(
                            OrderedDict(val_batch=per), refresh=True
                        )

                    outputs = {
                        key: outputs[key].cpu() if isinstance(outputs[key], Tensor) else outputs[key]
                        for key in outputs.keys()
                    }
                    self.experiment.log_metrics(outputs, step=self.global_step, epoch=epoch)
                    results.append(outputs)
                    # on_step_end()

                dic = self.val_end(results)        # pylint: disable=E1111

                if len(dic) != 0:
                    self._metrics.update(dic)
                    self.experiment.log_metrics(dic, step=epoch, epoch=epoch)

    def test_cycle(self, loader: DataLoader) -> None:
        """
        ニューラルネットの検証用ループです。

        Args:
            loader:

        Returns:

        """
        results = list()
        loader_size = len(loader)

        self.model.eval()
        with self.experiment.test():
            with no_grad():
                for step, batch in enumerate(loader):
                    batch = send(batch, self.device)
                    # on_step_start()
                    outputs = self.test_step(batch)        # pylint: disable=E1111

                    per = "{:1.0%}".format(step / loader_size)
                    if hasattr(self.pbar, "set_postfix"):
                        self.pbar.set_postfix(
                            OrderedDict(test_batch=per), refresh=True
                        )
                        self.pbar.update(1)

                    outputs = {
                        key: outputs[key].cpu() if isinstance(outputs[key], Tensor) else outputs[key]
                        for key in outputs.keys()
                    }

                    self.experiment.log_metrics(outputs)
                    results.append(outputs)
                    # on_step_end()

                dic = self.test_end(results)        # pylint: disable=E1111

                if len(dic) != 0:
                    self._metrics.update(dic)
                    self.experiment.log_metrics(dic)

    def train_config(self, epochs: int, checkpoint_path: Optional[str] = None, monitor: Optional[str] = None):
        """
        .run() メソッドを用いて訓練を行う際に、 epochs などを指定する為のメソッドです。

        Examples:
            >>> runner: BaseRunner = ...
            >>> runner.train_config(
            >>>     epochs=10,
            >>>     checkpoint_path="/path/to/checkpoint_dir",
            >>>     monitor="validate_avg_acc >= 0.75"
            >>> )

        Args:
            epochs (int): 訓練epoch数を指定する。
            checkpoint_path: checkpointを保存するディレクトリ名を指定。monitorを指定しなければ、全てのepochにおける重みを保存する。
            monitor: 指定した式に当てはまるepochのみを保存する。checkpoint_pathと一緒に設定する必要があるます。

        Returns:
            None
        Warnings:
            引数 monitor を指定する場合、「キーワード」「記号」「値」の間には必ずスペースを入れてください。

        """

        self.configures["checkpoint_path"]: str = checkpoint_path
        if monitor:
            try:
                _ = re.search("train|validate", monitor)[0]
            except TypeError:
                raise KeyError("The argument monitor is not an expected expression. {}".format(monitor))
            else:
                self.configures["monitor"] = monitor

            if not isinstance(self.experiment, Experiment):
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
        Runner の準備を行うメソッドです。
        self.model, self.optimizer, self.experiment といった実行に必要な最低限の変数が未定義な場合はエラーメッセージを出し終了します。
        問題がなければ CPU or GPU にモデルを渡します。

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

        if isinstance(self.experiment, Experiment):
            self.api_experiment = APIExperiment(previous_experiment=self.experiment.id, cache=False)

        if self.global_step < 0:
            self.global_step = 0

        self.model = self.model.to(self.device)

    def run(self, phase: str = "all", verbose: bool = True):
        """
        Runnerを実行します。
        実行には、事前に self.add_loader("train", train_loader) で訓練用のデータローダを登録するしておく必要があります。

        Args:
            phase:
                - `train`
                - `val`
                - `test`
                - `all`
                - `debug`

                のいずれかを指定してする事で、実行フェーズを決定します。デフォルト: all

            verbose: True の場合、学習の進行を表示します。

        Returns:
            None

        """
        phases = {"train", "train/val", "test", "all", "debug"}
        if phase not in phases:
            raise KeyError("The argument 'phase' must be one of the following. {}".format(phases))

        if phase == "debug":
            self.experiment.add_tag("debug")

        self.initialize()
        self.log_hyperparams()

        if not self.loaders:
            raise Exception("At least one DataLoader must be provided.")

        if phase in {"all", "train", "train/val", "debug"}:
            if "train" in self.loaders:
                self.pbar = tqdm(range(self.configures["epochs"]), desc="Epochs") if verbose \
                    else range(self.configures["epochs"])
                # .on_epoch_start()
                for epoch in self.pbar:
                    # on_train_start()
                    self.train_cycle(epoch, self.loaders["train"])
                    # on_train_end()

                    if phase in {"all", "train/val", "debug"}:
                        if "val" in self.loaders:
                            # on_validation_start()
                            self.val_cycle(epoch, self.loaders["val"])
                            # on_validation_end()

                    if self.scheduler:
                        self.update_scheduler(epoch)

                    if self.early_stop:
                        if self.early_stop.on_epoch_end(self._metrics, epoch):
                            break
                        # .on_epoch_end()

                    if self.configures["checkpoint_path"]:
                        ops = {
                            "==": operator.eq, "!=": operator.ne, "<": operator.lt,
                            "<=": operator.le, ">": operator.gt, ">=": operator.ge
                        }

                        if "monitor" in self.configures.keys():
                            key, op, value = self.configures["monitor"].split(" ")
                            value = float(value)

                            try:
                                current_value = float(self.api_experiment.get_metrics_summary(key)["valueCurrent"])
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
                # on_test_start()
                self.pbar = tqdm(total=len(self.loaders["test"]), desc="Evaluating") if verbose else None
                self.test_cycle(self.loaders["test"])
                # on_test_end()

        return self

    def predict(self, x: Union[Tensor, ndarray]) -> ndarray:
        """
        与えられた入力をもとに予測を行うメソッドです。

        Args:
            x (Union[torch.Tensor, np.ndarray]):

        Returns:
            predict
        """
        raise NotImplementedError

    def add_loader(self, mode: str, loader: DataLoader):
        """
        訓練等に用いるデータローダをRunnerに登録する為のメソッドです。

        Args:
            mode (str): ['train', 'val', 'test'] のいずれをか指定します。
            loader (torch.utils.data.DataLoader):

        Examples:
            >>> loader = DataLoader(...)
            >>> runner: BaseRunner = ...
            >>> runner.add_loader("train", loader)

        """
        if mode not in ["train", "val", "test"]:
            raise KeyError("argument `mode` must be one of 'train', 'val', or 'test'.")

        if not isinstance(loader, DataLoader):
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
        Scikit-Learn スタイルの訓練メソッドです。

        Args:
            x: 訓練用データ
            y: 教師ラベル
            **kwargs:

        """
        val_size: float = kwargs.get("val_size", 0.1)
        num_workers = kwargs.get("num_workers", 0)
        batch_size = kwargs.get("batch_size", 1)
        pin_memory = kwargs.get("pin_memory", False)
        verbose = kwargs.get("verbose", True)
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
            train_ds, batch_size=batch_size,
            sampler=train_sampler, num_workers=num_workers,
            pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size,
            sampler=val_sampler, num_workers=num_workers, pin_memory=pin_memory
        )

        self.add_loader("train", train_loader)
        self.add_loader("val", val_loader)
        self.train_config(epochs, checkpoint_path=checkpoint_path, monitor=monitor)
        self.run(verbose=verbose)

        return self

    def freeze(self) -> None:
        """
        モデルのパラメータが勾配を計算しないように固定する為のメソッドです。

        """
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

    def unfreeze(self) -> None:
        """
        .freeze() で固定されたパラメータを再度学習できるようにする為のメソッドです。

        """
        for param in self.model.parameters():
            param.requires_grad = True

        self.model.train()

    def save(self, directory: Optional[str] = None, epoch: Optional[int] = None) -> None:
        super(BaseRunner, self).save(directory, epoch)
