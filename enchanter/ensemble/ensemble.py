# *******************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# *******************************************************

from typing import List

import numpy as np
from torch.utils.data import Dataset
from sklearn.base import BaseEstimator

from enchanter.engine.runner import BaseRunner, ClassificationRunner
from enchanter.engine import modules

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

    def train(self, dataset: Dataset, epochs: int, batch_size: int, shuffle: bool = True, checkpoints: List[str] = None):
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

            runner.train(dataset, epochs, batch_size, shuffle, checkpoint)
            self.weights.append(runner.save_checkpoint()["model_state_dict"])

        return self

    def fit(self, x: np.ndarray, y: np.ndarray = None, **kwargs):
        epochs: int = kwargs.get("epochs", 1)
        batch_size: int = kwargs.get("batch_size", 1)
        checkpoint: List[str] = kwargs.get("checkpoint", None)

        train_ds = modules.get_dataset(x, y)
        self.train(train_ds, epochs, batch_size, checkpoints=checkpoint)
        return self

    def predict(self, x) -> List[np.ndarray]:
        x = modules.numpy2tensor(x)

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
