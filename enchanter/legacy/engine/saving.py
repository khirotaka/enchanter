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
        self.save_dir: Optional[str] = None

    def model_name(self) -> str:
        """
        fetch model name

        Returns: model name

        """
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
                - ``model_state_dict``: model weights
                - ``optimizer_state_dict``: Optimizer state

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
        Takes a dictionary with keys ``model_state_dict`` and ``optimizer_state_dict``
        and uses them to restore the state of the model and the Optimizer.

        Args:
            checkpoint:
                Takes a dictionary with the following keys and values.
                    - ``model_state_dict``: model weights
                    - ``optimizer_state_dict``: Optimizer state
        """
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return self

    def save(self, directory: Optional[str] = None, epoch: Optional[int] = None, filename: Optional[str] = None):

        """
        Save the model and the Optimizer state file in the specified directory.

        Notes:
            ``enchanter_checkpoints_epoch_{}.pth`` file contains ``model_state_dict`` & ``optimizer_state_dict``.

        Args:
            directory (Optional[str]):
            epoch (Optional[int]):
            filename (Optional[str]):

        """
        if directory is None and self.save_dir:
            directory = self.save_dir

        if directory is None:
            if filename is None:
                raise ValueError("The argument `directory` or `filename` must be specified.")
            else:
                path = filename
        else:
            directory_path = Path(directory)
            if not directory_path.exists():
                directory_path.mkdir(parents=True)

            if epoch is None:
                epoch_str = ctime().replace(" ", "_")
            else:
                epoch_str = str(epoch)

            if not filename:
                filename = "enchanter_checkpoints_epoch_{}.pth".format(epoch_str)

            path = str(directory_path / filename)

        checkpoint = self.save_checkpoint()
        torch.save(checkpoint, path)

        model_name = self.model_name()
        self.experiment.log_model(model_name, str(path))

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
