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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.experiment = None
        self.early_stop = None

        self._epochs = 1
        self.pbar = None
        self._loaders = {}
        self._metrics = {}

    @abstractmethod
    def train_step(self, batch):
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

    def val_step(self, batch):
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

    def test_step(self, batch):
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

    def train_cycle(self, epoch, loader):
        results = list()
        loader_size = len(loader)

        self.model.train()
        with self.experiment.train():
            for step, batch in enumerate(loader):
                self.optimizer.zero_grad()
                batch = tuple(map(lambda x: x.to(self.device), batch))
                # on_step_start()
                outputs = self.train_step(batch)
                outputs["loss"].backward()
                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()
                    self.experiment.log_metric("scheduler_lr", self.scheduler.get_last_lr(), step=step, epoch=epoch)

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

            dic = self.train_end(results)
            self._metrics.update(dic)
            self.experiment.log_metrics(dic, epoch=epoch)

    def val_cycle(self, epoch, loader):
        results = list()
        loader_size = len(loader)

        self.model.eval()
        with self.experiment.validate():
            with torch.no_grad():
                for step, batch in enumerate(loader):
                    batch = tuple(map(lambda x: x.to(self.device), batch))
                    # on_step_start()
                    outputs = self.val_step(batch)

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

                dic = self.val_end(results)
                self._metrics.update(dic)
                self.experiment.log_metrics(dic, epoch=epoch)

    def test_cycle(self, loader):
        results = list()
        loader_size = len(loader)

        self.model.eval()
        with self.experiment.test():
            with torch.no_grad():
                for step, batch in enumerate(loader):
                    batch = tuple(map(lambda x: x.to(self.device), batch))
                    # on_step_start()
                    outputs = self.test_step(batch)

                    per = "{:1.0%}".format(step / loader_size)
                    self.pbar.set_postfix(
                        OrderedDict(test_batch=per), refresh=True
                    )

                    outputs = {
                        key: outputs[key].cpu() if isinstance(outputs[key], torch.Tensor) else outputs[key]
                        for key in outputs.keys()
                    }

                    self.experiment.log_metrics(outputs, step=step)
                    results.append(outputs)
                    # on_step_end()

                dic = self.test_end(results)
                self._metrics.update(dic)
                self.experiment.log_metrics(dic)

    def train_config(self, epochs, *args, **kwargs):
        if epochs > 0:
            self._epochs = epochs

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

    def standby(self):
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

                if self.early_stop:
                    if self.early_stop.on_epoch_end(self._metrics, epoch):
                        break
                    # .on_epoch_end()

        if "test" in self.loaders:
            # on_test_start()
            self.test_cycle(self.loaders["test"])
            # on_test_end()

        return self

    def predict(self, x):
        """

        Args:
            x (Union[torch.Tensor, np.ndarray]):

        Returns:
            predict
        """

    def add_loader(self, mode, loader):
        """

        Args:
            mode (str):
            loader (torch.utils.data.DataLoader):

        Returns:

        """
        if mode not in ["train", "val", "test"]:
            raise Exception("argument `mode` must be one of 'train', 'val', or 'test'.")

        if not isinstance(loader, torch.utils.data.DataLoader):
            raise Exception("The argument `loader` must be an instance of `torch.utils.data.DataLoader`.")

        self.experiment.log_parameters(loader.__dict__, prefix=mode)
        self.experiment.log_parameter("{}_dataset_len".format(mode), len(loader))

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

    def load_checkpoint(self, checkpoint):
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return self

    def save(self, directory, epoch=None) -> None:
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

    def load(self, filename, map_location="cpu"):
        """
        Args:
            filename:
            map_location:
        Returns:
        """
        checkpoint = torch.load(filename, map_location=map_location)
        self.load_checkpoint(checkpoint)

        return self
