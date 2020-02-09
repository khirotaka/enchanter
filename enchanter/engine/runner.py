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
from typing import Dict, Union, Any, List

import torch
import numpy as np
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score

from . import modules

if modules.is_jupyter():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class BaseRunner(BaseEstimator):
    def __init__(self, model, criterion, optimizer, optim_config, device=None, experiment=None, scheduler=None):
        """

        Args:
            model (nn.Module):
            criterion:
            optimizer:
            optim_config (dict):
            device:
            experiment:
            scheduler (Dict[str, Union]):
        """
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: torch.nn.Module = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer(self.model.parameters(), **optim_config)
        self.logger = modules.CometLogger(experiment) if experiment else None
        self.scheduler = scheduler["algorithm"](self.optimizer, **scheduler["config"]) if scheduler else None

    def forward(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        順伝搬・損失値の計算を行うメソッド。
        算出した loss を返します。

        Args:
            data: Training Data.
            target: Training Label for supervised learning.

        Returns:
            loss: loss value which calculated by self.criterion.
        """
        data = data.to(self.device)
        target = target.to(self.device)

        out = self.model(data)
        loss = self.criterion(out, target)
        return loss

    def validate(self, data: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        validationを行うメソッド。
        各算出した値を格納した辞書を返します。

        Args:
            data: Validating Data.
            target: Validating Label for supervised learning.

        Returns:
            results: dictionary which contained some results.
        """
        results = {}

        loss = self.forward(data, target)
        results["loss"] = loss.cpu()

        return results

    def train(self, dataset, epochs, batch_size, verbose=True, shuffle=False, checkpoint=False, validation=None, **loader_config):
        """

        Args:
            dataset (Dataset):
            epochs (int):
            batch_size (int):
            verbose (bool):
            shuffle (bool):
            checkpoint (str):
            validation (Dict[str, Union[Dataset, Dict]]):

        Examples:
            >>> ds: Dataset = MNIST( ... )
            >>> model: torch.nn.Module = torch.nn.Sequential(
            >>>     ...
            >>> )
            >>> runner = ClassificationRunner(model, torch.nn.NLLLoss(), torch.optim.SGD, {"lr": 0.001})
            >>> runner.train(ds, epochs=10, batch_size=32, shuffle=True)

        Returns:

        """

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **loader_config)

        val_loader = DataLoader(
            validation["dataset"], batch_size=batch_size, shuffle=shuffle, **validation["config"]
        ) if validation else None

        epoch_bar = tqdm(range(epochs), desc="Epochs", leave=True) if verbose else range(epochs)
        for epoch in epoch_bar:
            self.model.train()
            step_bar = tqdm(train_loader, desc="Training", leave=False) if verbose else train_loader
            for i, (x, y) in enumerate(step_bar):

                self.optimizer.zero_grad()
                loss = self.forward(x, y)

                loss.backward()
                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                if self.logger:
                    self.logger.log_train(epoch, i, {"loss": loss.detach().cpu()})

                    if self.scheduler:
                        self.logger.log_train(epoch, i, {"lr": self.scheduler.get_last_lr()})

            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    val_bar = tqdm(val_loader, desc="Validation", leave=False) if verbose else val_loader
                    for j, (x_val, y_val) in enumerate(val_bar):
                        val_results = self.validate(x_val, y_val)

                        if self.logger:
                            self.logger.log_val(epoch, j, val_results)

            if checkpoint:
                self.save(checkpoint, epoch=epoch+1)

        return self

    def fit(self, x: np.ndarray, y: np.ndarray = None, **kwargs):
        """
        sklearn API


        Args:
            x:
            y:
            **kwargs:

        Returns:

        """
        epochs: int = kwargs.get("epochs", 1)
        batch_size: int = kwargs.get("batch_size", 1)
        checkpoint: str = kwargs.get("checkpoint", None)
        pin_memory: bool = kwargs.get("pin_memory", False)
        train_rate: float = kwargs.get("train_rate", 0.8)
        val_ds = kwargs.get("val_ds", None)
        verbose: bool = kwargs.get("verbose", True)
        loader_config: Dict = kwargs.get("loader_config", dict())

        train_ds = modules.get_dataset(x, y)

        if not val_ds:
            n_samples = len(train_ds)
            train_size = int(n_samples * train_rate)
            val_size = n_samples - train_size
            train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])

        self.train(
            dataset=train_ds,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            checkpoint=checkpoint,
            pin_memory=pin_memory,
            validation={
                "dataset": val_ds,
                "config": {
                    "pin_memory": pin_memory
                }
            },
            **loader_config
        )
        return self

    def predict(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """

        Args:
            x (Union[torch.Tensor, np.ndarray]):

        Examples:
            >>> x = mnist_img    # [1, 1, 28, 28]
            >>> prediction = runner.predict(x)

        Returns:
            prediction (np.ndarray)
        """
        x = modules.numpy2tensor(x)
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            out = self.model(x).cpu().numpy()
        return out

    def evaluate(self, x: Any, y: Any, batch_size: int = 1):
        raise NotImplementedError

    def save_checkpoint(self) -> dict:
        checkpoint = {
            "model_state_dict": deepcopy(self.model.state_dict()),
            "optimizer_state_dict": deepcopy(self.optimizer.state_dict())
        }
        return checkpoint

    def load_checkpoint(self, checkpoint: dict):
        """

        Args:
            checkpoint:

        Returns:

        """
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def save(self, directory: str, epoch: int = None) -> None:
        """

        Args:
            directory:
            epoch:

        Returns:

        """
        if not os.path.isdir(directory):
            os.mkdir(directory)
        checkpoint = self.save_checkpoint()

        if epoch is None:
            epoch = time.ctime().replace(" ", "_")

        filename = "checkpoint_epoch_{}.pth".format(epoch)
        path = directory + filename
        torch.save(checkpoint, path)

        if self.logger:
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            self.logger.experiment.log_asset_data(buffer.getvalue(), filename)

    def load(self, filename: str, map_location: str = "cpu") -> None:
        """

        Args:
            filename:
            map_location:

        Returns:

        """
        checkpoint = torch.load(filename, map_location=map_location)
        self.load_checkpoint(checkpoint)


class ClassificationRunner(BaseRunner):
    def predict(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """

        Args:
            x:

        Returns:

        """
        out = super(ClassificationRunner, self).predict(x)
        predict = np.argmax(out, axis=-1)
        return predict

    def evaluate(self, x, y=None, batch_size: int = 1, verbose: bool = True, metrics: List[callable] = None) -> Dict:
        """

        Args:
            x (Union[Union[np.ndarray, torch.Tensor], Dataset]):
            y (Union[Union[np.ndarray, torch.Tensor], None]):
            batch_size (int):
            verbose (bool):
            metrics (List[callable]): sklearn.metrics のような、 func(y_true, y_pred) の形で提供される評価関数を格納した配列

        Returns:
            losses (float):
            accuracy (float):
        """
        if metrics is None:
            metrics = [accuracy_score]
        else:
            metrics.append(accuracy_score)

        total = 0.0
        losses = 0.0

        metric_values = dict()

        predicts = []
        labels = []

        if x is not Dataset and y is not None:
            x = modules.get_dataset(x, y)

        loader = tqdm(DataLoader(x, batch_size=batch_size, shuffle=False), desc="Evaluating")\
            if verbose else DataLoader(x, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for x, y in loader:
                total += y.shape[0]

                x = x.to(self.device)
                y = y.to(self.device)

                loss = self.criterion(self.model(x), y).cpu().item()
                predict = self.predict(x)
                losses += loss

                labels.append(y.cpu().numpy())
                predicts.append(predict)

        labels = np.vstack(labels)
        predicts = np.vstack(predicts)

        for func in metrics:
            metric_values[func.__name__] = func(labels, predicts)

        metric_values["loss"] = losses / total

        if self.logger:
            self.logger.log_test(metric_values)

        return metric_values
