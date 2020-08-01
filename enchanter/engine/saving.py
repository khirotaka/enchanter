from typing import Union, Optional, Dict
from collections import OrderedDict
from time import ctime
from pathlib import Path
from copy import deepcopy

from torch.tensor import Tensor
from torch.nn import DataParallel
from torch.serialization import save, load


__all__ = [
    "RunnerIO"
]


class RunnerIO:
    """
    PyTorch モデルの重み、Optimizerの状態といったパラメータの読み込み・保存を担当するクラス。

    """
    def __init__(self):
        self.model = NotImplemented
        self.optimizer = NotImplemented
        self.experiment = NotImplemented
        self._checkpoint_path = NotImplemented

    def save_checkpoint(self) -> Dict[str, Union[Dict[str, Tensor], dict]]:
        """
        ニューラルネットの重みと、 Optimizerの状態を辞書として出力するメソッドです。

        Returns:
            以下のキーと値を持つ辞書を返します。
                - "model_state_dict": ニューラルネットの重み
                - "optimizer_state_dict": Optimizer の状態

        """
        if isinstance(self.model, DataParallel):
            model = self.model.module.state_dict()
        else:
            model = self.model.state_dict()

        checkpoint = {
            "model_state_dict": deepcopy(model),
            "optimizer_state_dict": deepcopy(self.optimizer.state_dict())
        }
        return checkpoint

    def load_checkpoint(self, checkpoint: Dict[str, OrderedDict]):
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

    def save(self, directory: Optional[str] = None, epoch: Optional[int] = None):

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
            epoch = ctime().replace(" ", "_")

        filename = "checkpoint_epoch_{}.pth".format(epoch)
        path = directory / filename
        save(checkpoint, path)

        if hasattr(self.experiment, "log_asset"):
            self.experiment.log_asset(path)

    def load(self, filename: str, map_location: str = "cpu"):
        """
        指定したファイルを元にモデルとOptimizerの状態を復元します。

        Args:
            filename (str):
            map_location (str):

        """
        checkpoint = load(filename, map_location=map_location)
        self.load_checkpoint(checkpoint)

        return self
