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
import time
from copy import deepcopy
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
from sklearn import base
from torch.utils.data import DataLoader

from enchanter.engine import modules

if modules.is_jupyter():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class BaseRunner(base.BaseEstimator, ABC):
    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.experiment = None

        self._epochs = 1

        self._loaders = {}

    @abstractmethod
    def train_step(self, batch) -> Dict[str, torch.Tensor]:
        """

        Args:
            batch:

        Returns:

        """

    def train_end(self, outputs):
        """

        Args:
            outputs:

        Returns:

        """

    def val_step(self, batch) -> Dict[str, torch.Tensor]:
        """

        Args:
            batch:

        Returns:

        """

    def val_end(self, outputs):
        """

        Args:
            outputs:

        Returns:

        """

    def test_step(self, batch) -> Dict[str, torch.Tensor]:
        """

        Args:
            batch:

        Returns:

        """

    def test_end(self, outputs):
        """

        Args:
            outputs:

        Returns:

        """

    def train_cycle(self, epoch, loader) -> None:
        self.model.train()
        with self.experiment.train():
            for step, batch in enumerate(loader):
                self.optimizer.zero_grad()
                batch = tuple(map(lambda x: x.to(self.device), batch))
                outputs = self.train_step(batch)
                outputs["loss"].backward()
                self.optimizer.step()

                self.experiment.log_metrics(outputs, step=step, epoch=epoch)
            self.train_end(outputs)

    def val_cycle(self, epoch, loader) -> None:
        self.model.eval()
        with self.experiment.validate():
            with torch.no_grad():
                for step, batch in enumerate(loader):
                    batch = tuple(map(lambda x: x.to(self.device), batch))
                    outputs = self.val_step(batch)
                    self.experiment.log_metrics(outputs, step=step, epoch=epoch)
                self.val_end(outputs)

    def test_cycle(self, loader) -> None:
        self.model.eval()
        with self.experiment.test():
            with torch.no_grad():
                for step, batch in enumerate(loader):
                    batch = tuple(map(lambda x: x.to(self.device), batch))
                    outputs = self.test_step(batch)
                    self.experiment.log_metrics(outputs, step=step)
                self.test_end(outputs)

    def train_config(self, epochs, *args, **kwargs):
        self._epochs = epochs

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device

    def log_hyperparams(self, dic=None, prefix=None) -> None:
        """

        Args:
            dic (Dict):
            prefix (str):

        Returns:
            None
        """
        if hasattr(self.experiment, "set_model_graph"):
            self.experiment.set_model_graph(self.model.__repr__())

        self.experiment.log_parameters(self.optimizer.__dict__["defaults"], prefix="optimizer")
        self.experiment.log_parameter("Optimizer", self.optimizer.__class__.__name__)

        if dic is not None:
            self.experiment.log_parameters(dic, prefix)

    def standby(self) -> None:
        if self.model is None:
            raise Exception("self.model is not defined.")

        if self.optimizer is None:
            raise Exception("self.optimizer is not defined.")

        if self.experiment is None:
            raise Exception("self.experiment is not defined.")

        self.model = self.model.to(self.device)

    def run(self, verbose=True):
        self.log_hyperparams()
        self.standby()

        if not self.loaders:
            raise Exception("At least one DataLoader must be provided.")

        if "train" in self.loaders:
            pbar = tqdm(range(self._epochs), desc="Epochs") if verbose else range(self._epochs)
            for epoch in pbar:
                self.train_cycle(epoch, self.loaders["train"])

                if "val" in self.loaders:
                    self.val_cycle(epoch, self.loaders["val"])

        if "test" in self.loaders:
            self.test_cycle(self.loaders["test"])

        return self

    def add_loader(self, loader: torch.utils.data.DataLoader, mode):
        """

        Args:
            loader (torch.utils.data.DataLoader):
            mode (str):

        Returns:

        """
        if mode not in ["train", "val", "test"]:
            raise Exception("argument `mode` must be one of 'train', 'val', or 'test'.")

        if not isinstance(loader, torch.utils.data.DataLoader):
            raise Exception("The argument `loader` must be an instance of` torrch.utils.data.Dataloader`.")

        self._loaders[mode] = loader
        return self

    @property
    def loaders(self):
        return self._loaders

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

        self.model.train()

    def save_checkpoint(self) -> Dict[str, OrderedDict]:
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module.state_dict()
        else:
            model = self.model.state_dict()

        checkpoint = {
            "model_state_dict": deepcopy(model),
            "optimizer_state_dict": deepcopy(self.optimizer.state_dict())
        }
        return checkpoint

    def load_checkpoint(self, checkpoint: Dict[str, OrderedDict]):
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return self

    def save(self, directory: str, epoch: int = None) -> None:
        """
        Args:
            directory:
            epoch:
        Returns:
        """
        if not os.path.isdir(directory):
            os.makedirs(directory)
        checkpoint = self.save_checkpoint()

        if epoch is None:
            epoch = time.ctime().replace(" ", "_")

        filename = "checkpoint_epoch_{}.pth".format(epoch)
        path = directory + filename
        torch.save(checkpoint, path)

        if hasattr(self.experiment, "log_asset_data"):
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            self.experiment.log_asset_data(buffer.getvalue(), filename)

    def load(self, filename: str, map_location: str = "cpu"):
        """
        Args:
            filename:
            map_location:
        Returns:
        """
        checkpoint = torch.load(filename, map_location=map_location)
        self.load_checkpoint(checkpoint)

        return self
