# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

import io
import time
from pathlib import Path
from copy import deepcopy
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from sklearn import base
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

from enchanter.engine import modules

if modules.is_jupyter():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


__all__ = [
    "BaseRunner"
]


class BaseRunner(base.BaseEstimator, ABC):
    """
    PyTorchモデルの訓練に用いる Runner を作成する為のクラスです。


    Examples:

        >>> class Runner(BaseRunner):
        >>>     def __init__(self):
        >>>         super(Runner, self).__init__()
        >>>         self.model = nn.Linear(10, 10)
        >>>         self.optimizer = torch.optim.Adam(self.model.parameters())
        >>>         self.experiment = Experiment()
        >>>         self.criterion = nn.CrossEntropyLoss()
        >>>
        >>>     def train_step(self, batch):
        >>>         x, y = batch
        >>>         out = self.model(x)
        >>>         loss = self.criterion(out, y)
        >>>
        >>>         return {"loss": loss}

    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NotImplemented
        self.optimizer = NotImplemented
        self.scheduler = None
        self.experiment = NotImplemented
        self.early_stop = None

        self._epochs = 1
        self.pbar = None
        self._loaders = {}
        self._metrics = {}
        self._checkpoint_path = None

    @abstractmethod
    def train_step(self, batch):
        """
        ニューラルネットの訓練時、
            >>> for x, y in train_loader:
            >>>     out = model(x)
            >>>     loss = criteion(out, y)

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

    def train_end(self, outputs):
        """
        ニューラルネットの訓練時、1step 終了ごとに実行されるメソッドです。

        Args:
            outputs:

        Returns:

        """
        return {}

    def val_step(self, batch):
        """
        ニューラルネットの検証時、1step ごとに実行されるメソッドです。

        Args:
            batch: PyTorch DataLoader から得られる訓練用のデータトラベルを含むタプル

        Returns:
            辞書を返す必要があります。

        """

    def val_end(self, outputs):
        """
        ニューラルネットの検証時、1step 終了ごとに実行されるメソッドです。

        Args:
            outputs:

        Returns:

        """
        return {}

    def test_step(self, batch):
        """
        ニューラルネットの評価時、1step ごとに実行されるメソッドです。

        Args:
            batch: PyTorch DataLoader から得られる訓練用のデータトラベルを含むタプル

        Returns:
            辞書を返す必要があります。

        """

    def test_end(self, outputs):
        """
        ニューラルネットの評価時、1step 終了ごとに実行されるメソッドです。

        Args:
            outputs:

        Returns:

        """
        return {}

    def train_cycle(self, epoch, loader):
        """
        ニューラルネットの訓練ループです。

        Args:
            epoch (int):
            loader (torch.utils.data.DataLoader):

        """
        results = list()
        loader_size = len(loader)

        self.model.train()
        with self.experiment.train():
            for step, batch in enumerate(loader):
                self.optimizer.zero_grad()
                batch = modules.send(batch, self.device)
                # on_step_start()
                outputs = self.train_step(batch)
                outputs["loss"].backward()
                self.optimizer.step()

                if hasattr(self.pbar, "set_postfix"):
                    per = "{:1.0%}".format(step / loader_size)
                    self.pbar.set_postfix(
                        OrderedDict(train_batch=per), refresh=True
                    )

                outputs = {
                    key: outputs[key].detach().cpu() if isinstance(outputs[key], torch.Tensor) else outputs[key]
                    for key in outputs.keys()
                }
                self.experiment.log_metrics(outputs, step=step, epoch=epoch)
                results.append(outputs)
                # on_step_end()

            dic = self.train_end(results)        # pylint: disable=E1111

            if len(dic) != 0:
                self._metrics.update(dic)
                self.experiment.log_metrics(dic, step=epoch)

    def val_cycle(self, epoch, loader):
        """
        ニューラルネットの評価用ループです。

        Args:
            epoch:
            loader:

        Returns:

        """
        results = list()
        loader_size = len(loader)

        self.model.eval()
        with self.experiment.validate():
            with torch.no_grad():
                for step, batch in enumerate(loader):
                    batch = modules.send(batch, self.device)
                    # on_step_start()
                    outputs = self.val_step(batch)        # pylint: disable=E1111

                    if hasattr(self.pbar, "set_postfix"):
                        per = "{:1.0%}".format(step / loader_size)
                        self.pbar.set_postfix(
                            OrderedDict(val_batch=per), refresh=True
                        )

                    outputs = {
                        key: outputs[key].cpu() if isinstance(outputs[key], torch.Tensor) else outputs[key]
                        for key in outputs.keys()
                    }
                    self.experiment.log_metrics(outputs, step=step, epoch=epoch)
                    results.append(outputs)
                    # on_step_end()

                dic = self.val_end(results)        # pylint: disable=E1111

                if len(dic) != 0:
                    self._metrics.update(dic)
                    self.experiment.log_metrics(dic, step=epoch)

    def test_cycle(self, loader):
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
            with torch.no_grad():
                for step, batch in enumerate(loader):
                    batch = modules.send(batch, self.device)
                    # on_step_start()
                    outputs = self.test_step(batch)        # pylint: disable=E1111

                    per = "{:1.0%}".format(step / loader_size)
                    if hasattr(self.pbar, "set_postfix"):
                        self.pbar.set_postfix(
                            OrderedDict(test_batch=per), refresh=True
                        )
                        self.pbar.update(1)

                    outputs = {
                        key: outputs[key].cpu() if isinstance(outputs[key], torch.Tensor) else outputs[key]
                        for key in outputs.keys()
                    }

                    self.experiment.log_metrics(outputs, step=step)
                    results.append(outputs)
                    # on_step_end()

                dic = self.test_end(results)        # pylint: disable=E1111

                if len(dic) != 0:
                    self._metrics.update(dic)
                    self.experiment.log_metrics(dic)

    def train_config(self, epochs, **kwargs):
        """
        .run() メソッドを用いて訓練を行う際に、 epochs などを指定する為のメソッドです。

        Args:
            epochs (int):
            **kwargs:

        Returns:

        """

        self._checkpoint_path = kwargs.get("checkpoint_path", None)

        if epochs > 0:
            self._epochs = epochs
        else:
            self._epochs = 1

        return self

    def log_hyperparams(self, dic=None, prefix=None):
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

    def initialize(self):
        """
        Runner の準備を行うメソッドです。
        self.model, self.optimizer, self.experiment といった実行に必要な最低限の変数が未定義な場合はエラーメッセージを出し終了します。
        問題がなければ CPU or GPU にモデルを渡します。

        Returns:
            None
        """
        if self.model is None:
            raise Exception("self.model is not defined.")

        if self.optimizer is None:
            raise Exception("self.optimizer is not defined.")

        if self.experiment is None:
            raise Exception("self.experiment is not defined.")

        self.model = self.model.to(self.device)

    def run(self, verbose=True):
        """
        Runnerを実行します。
        実行には、事前に self.add_loader("train", train_loader) で訓練用のデータローダを登録するしておく必要があります。

        Args:
            verbose: True の場合、学習の進行を表示します。

        Returns:
            None
        """
        self.initialize()
        self.log_hyperparams()

        if not self.loaders:
            raise Exception("At least one DataLoader must be provided.")

        if "train" in self.loaders:
            self.pbar = tqdm(range(self._epochs), desc="Epochs") if verbose else range(self._epochs)
            # .on_epoch_start()
            for epoch in self.pbar:
                # on_train_start()
                self.train_cycle(epoch, self.loaders["train"])
                # on_train_end()

                if "val" in self.loaders:
                    # on_validation_start()
                    self.val_cycle(epoch, self.loaders["val"])
                    # on_validation_end()

                if self.scheduler:
                    self.scheduler.step(epoch=None)
                    self.experiment.log_metric("scheduler_lr", self.scheduler.get_lr(), epoch=epoch)

                if self.early_stop:
                    if self.early_stop.on_epoch_end(self._metrics, epoch):
                        break
                    # .on_epoch_end()

                if self._checkpoint_path:
                    self.save(self._checkpoint_path, epoch=epoch)

        if "test" in self.loaders:
            # on_test_start()
            self.pbar = tqdm(total=len(self.loaders["test"]), desc="Evaluating") if verbose else None
            self.test_cycle(self.loaders["test"])
            # on_test_end()

        return self

    def predict(self, x):
        """
        与えられた入力をもとに予測を行うメソッドです。

        Args:
            x (Union[torch.Tensor, np.ndarray]):

        Returns:
            predict
        """
        raise NotImplementedError

    def add_loader(self, mode, loader):
        """
        訓練等に用いるデータローダをRunnerに登録する為のメソッドです。

        Args:
            mode (str): ['train', 'val', 'test'] のいずれをか指定します。
            loader (torch.utils.data.DataLoader):

        """
        if mode not in ["train", "val", "test"]:
            raise Exception("argument `mode` must be one of 'train', 'val', or 'test'.")

        if not isinstance(loader, DataLoader):
            raise Exception("The argument `loader` must be an instance of `torch.utils.data.DataLoader`.")

        self._loaders[mode] = loader
        return self

    @property
    def loaders(self):
        return self._loaders

    def fit(self, x, y, **kwargs):
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

        if self._epochs == 0:
            epochs = kwargs.get("epochs", 1)
        else:
            epochs = self._epochs

        train_ds = modules.get_dataset(x, y)
        val_ds = modules.get_dataset(x, y)
        n_train = len(train_ds)
        indices = list(range(n_train))
        split = int(np.floor(val_size * n_train))

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
        self.train_config(epochs, checkpoint_path=self._checkpoint_path)
        self.run(verbose)

        return self

    def freeze(self):
        """
        モデルのパラメータが勾配を計算しないように固定する為のメソッドです。

        """
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

    def unfreeze(self):
        """
        .freeze() で固定されたパラメータを再度学習できるようにする為のメソッドです。

        """
        for param in self.model.parameters():
            param.requires_grad = True

        self.model.train()

    def save_checkpoint(self):
        """
        ニューラルネットの重みと、 Optimizerの状態を辞書として出力するメソッドです。

        Returns:
            以下のキーと値を持つ辞書を返します。
                - "model_state_dict": ニューラルネットの重み
                - "optimizer_state_dict": Optimizer の状態

        """
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module.state_dict()
        else:
            model = self.model.state_dict()

        checkpoint = {
            "model_state_dict": deepcopy(model),
            "optimizer_state_dict": deepcopy(self.optimizer.state_dict())
        }
        return checkpoint

    def load_checkpoint(self, checkpoint):
        """
        'model_state_dict', 'optimizer_state_dict' を持つ辞書を受け取り、それらを元に モデル と Optimizer の状態を復元します。

        Args:
            checkpoint:
                以下のキーと値を持つ辞書。
                    - "model_state_dict": ニューラルネットの重み
                    - "optimizer_state_dict": Optimizer の状態
        """
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return self

    def save(self, directory=None, epoch=None):

        """
        指定したディレクトリにモデルとOptimizerの状態を記録したファイルを保存します。

        Args:
            directory (Optional[str]):
            epoch (Optional[int]):

        """
        if directory is None and self._checkpoint_path is not None:
            directory = self._checkpoint_path

        elif directory is None and self._checkpoint_path is None:
            raise ValueError("The argument `directory` must be specified.")

        directory = Path(directory)
        if not directory.exists():
            directory.mkdir(parents=True)
        checkpoint = self.save_checkpoint()

        if epoch is None:
            epoch = time.ctime().replace(" ", "_")

        filename = "checkpoint_epoch_{}.pth".format(epoch)
        path = directory / filename
        torch.save(checkpoint, path)

        if hasattr(self.experiment, "log_asset_data"):
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            self.experiment.log_asset_data(buffer.getvalue(), filename)

    def load(self, filename, map_location="cpu"):
        """
        指定したファイルを元にモデルとOptimizerの状態を復元します。

        Args:
            filename (str):
            map_location (str):

        """
        checkpoint = torch.load(filename, map_location=map_location)
        self.load_checkpoint(checkpoint)

        return self
