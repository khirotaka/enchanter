from typing import List

import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.base import BaseEstimator

from .runner import BaseRunner, ClassificationRunner
from . import modules

if modules.is_jupyter():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class BaseEnsembleEstimator(BaseEstimator):
    def __init__(self, runners: List[BaseRunner], mode: str = None):
        self.runners: List[BaseRunner] = runners
        self.weights: List = []
        self.mode: str = mode
        self.do_fit: bool = False

    def fit(self, dataset: Dataset, epochs: int, batch_size: int, shuffle: bool = True, checkpoints: List[str] = None):
        """

        Args:
            dataset:
            epochs:
            batch_size:
            shuffle:
            checkpoints:

        Returns:

        """
        self.do_fit = True
        for i, runner in enumerate(tqdm(self.runners, desc="Runner")):
            checkpoint = checkpoints[i] if checkpoints else None

            runner.fit(dataset, epochs, batch_size, shuffle, checkpoint)
            self.weights.append(runner.save_checkpoint()["model_state_dict"])

        return self

    def predict(self, x) -> List[np.ndarray]:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        predicts = []
        for i, runner in enumerate(self.runners):
            if self.do_fit:
                runner.model.load_state_dict(self.weights[i])
            predicts.append(runner.predict(x))

        return predicts


class SoftEnsemble(BaseEnsembleEstimator):
    """
    確率の平均をとるアンサンブル
    """

    def predict(self, x) -> np.ndarray:
        """

        Args:
            x:

        Returns:

        """
        predicts = super().predict(x)

        predicts = sum(predicts)
        probs = predicts / len(self.runners)

        if self.mode == "classification":
            return probs.astype(np.int)
        else:
            return probs


class HardEnsemble(BaseEnsembleEstimator):
    """
    多数決をとるアンサンブル
    """
    def __init__(self, runners: List[ClassificationRunner]):
        super().__init__(runners, mode="classification")

    def predict(self, x) -> np.ndarray:
        """

        Args:
            x:

        Returns:

        """
        predicts = super().predict(x)
        predicts = np.stack(predicts)

        return np.ravel(predicts[0])
