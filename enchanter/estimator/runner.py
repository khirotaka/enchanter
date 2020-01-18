import os
import time
from typing import Tuple
from copy import deepcopy

import torch
import numpy as np
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader, Dataset

from . import modules

if modules.is_jupyter():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class BaseRunner(BaseEstimator):
    def __init__(self, model, criterion, optimizer, optim_conf: dict, device: str = None) -> None:
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer(self.model.parameters(), **optim_conf)

    def one_cycle(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        data = data.to(self.device)
        target = target.to(self.device)

        self.optimizer.zero_grad()
        out = self.model(data)
        loss = self.criterion(out, target)
        return loss

    def fit(self, dataset: Dataset, epochs: int, batch_size: int, shuffle: bool = True, checkpoint: str = False):
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        for epoch in tqdm(range(epochs), desc="Epochs", leave=True):
            self.model.train()
            for x, y in tqdm(train_loader, desc="Training", leave=False):
                loss = self.one_cycle(x, y)
                loss.backward()
                self.optimizer.step()

            if checkpoint:
                self.save(checkpoint, epoch=epoch+1)

        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            out = self.model(x).cpu()
        return out

    def evaluate(self, dataset: Dataset):
        raise NotImplementedError

    def save_checkpoint(self) -> dict:
        checkpoint = {
            "model_state_dict": deepcopy(self.model.state_dict()),
            "optimizer_state_dict": deepcopy(self.optimizer.state_dict())
        }
        return checkpoint

    def load_checkpoint(self, checkpoint: dict):
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def save(self, directory: str, epoch: int = None) -> None:
        if not os.path.isdir(directory):
            os.mkdir(directory)
        checkpoint = self.save_checkpoint()

        if epoch is None:
            epoch = time.ctime().replace(" ", "_")

        filename = "checkpoint_epoch_{}.pth".format(epoch)
        filename = directory + filename
        torch.save(checkpoint, filename)

    def load(self, filename: str, map_location: str = "cpu") -> None:
        checkpoint = torch.load(filename, map_location=map_location)
        self.load_checkpoint(checkpoint)


class ClassificationRunner(BaseRunner):
    def predict(self, x: torch.Tensor) -> np.ndarray:
        out = super(ClassificationRunner, self).predict(x)
        _, predict = torch.max(out, dim=1)
        return predict.numpy()

    def evaluate(self, dataset: Dataset) -> Tuple[float, float]:
        correct = 0.0
        total = 0.0
        losses = 0.0

        loader = DataLoader(dataset)
        with torch.no_grad():
            for x, y in tqdm(loader, desc="Evaluating"):
                total += y.shape[0]

                out = self.model(x)
                loss = self.criterion(out, y).cpu().item()
                predict = self.predict(x)
                correct += np.sum(predict == y.cpu().numpy()).item()
                losses += loss

        return losses / total, correct / total
