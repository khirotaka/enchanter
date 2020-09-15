from typing import Union, Optional, Dict
from collections import OrderedDict
from time import ctime
from pathlib import Path
from copy import deepcopy

from torch.tensor import Tensor
from torch.nn import DataParallel
from torch.serialization import save, load


__all__ = ["RunnerIO"]


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
            "optimizer_state_dict": deepcopy(self.optimizer.state_dict()),
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
        Save the model and the Optimizer state file in the specified directory.

        Notes:
            ``enchanter_checkpoints_epoch_{}.pth`` file contains ``model_state_dict`` & ``optimizer_state_dict``.

        Args:
            directory (Optional[str]):
            epoch (Optional[int]):

        """
        if directory is None:
            if self._checkpoint_path is not None:
                directory_name: str = self._checkpoint_path
            else:
                raise ValueError("The argument `directory` must be specified.")
        else:
            directory_name = directory

        directory_path = Path(directory_name)
        if not directory_path.exists():
            directory_path.mkdir(parents=True)
        checkpoint = self.save_checkpoint()

        if epoch is None:
            epoch_str = ctime().replace(" ", "_")
        else:
            epoch_str = str(epoch)

        filename = "enchanter_checkpoints_epoch_{}.pth".format(epoch_str)
        path = directory_path / filename
        save(checkpoint, path)

        if hasattr(self.experiment, "log_model"):
            model_name = self.model.__class__.__name__
            self.experiment.log_model(model_name, str(path))

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
