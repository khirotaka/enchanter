from typing import Union, Optional, Dict
from collections import OrderedDict
from time import ctime
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn


__all__ = ["RunnerIO"]


class RunnerIO:
    """
    A class responsible for loading and saving parameters such as PyTorch model weights and Optimizer state.

    """

    def __init__(self):
        self.model = NotImplemented
        self.optimizer = NotImplemented
        self.experiment = NotImplemented
        self._checkpoint_path = NotImplemented

    def model_name(self) -> str:
        if isinstance(self.model, nn.DataParallel) or isinstance(self.model, nn.parallel.DistributedDataParallel):
            model_name = self.model.module.__class__.__name__
        else:
            model_name = self.model.__class__.__name__

        return model_name

    def save_checkpoint(self) -> Dict[str, Union[Dict[str, torch.Tensor], dict]]:
        """
        A method to output model weights and Optimizer state as a dictionary.

        Returns:
            Returns a dictionary with the following keys and values.
                - "model_state_dict": model weights
                - "optimizer_state_dict": Optimizer state

        """
        if isinstance(self.model, nn.DataParallel) or isinstance(self.model, nn.parallel.DistributedDataParallel):
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
        Takes a dictionary with keys 'model_state_dict' and 'optimizer_state_dict'
        and uses them to restore the state of the model and the Optimizer.

        Args:
            checkpoint:
                Takes a dictionary with the following keys and values.
                    - "model_state_dict": model weights
                    - "optimizer_state_dict": Optimizer state
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
        torch.save(checkpoint, path)

        if hasattr(self.experiment, "log_model"):
            self.experiment.log_model(self.model_name(), str(path))

    def load(self, filename: str, map_location: str = "cpu"):
        """
        Restores the model and Optimizer state based on the specified file.

        Args:
            filename (str):
            map_location (str): default: 'cpu'

        """
        checkpoint = torch.load(filename, map_location=map_location)
        self.load_checkpoint(checkpoint)

        return self
