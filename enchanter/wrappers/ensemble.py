# *******************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# *******************************************************

from typing import List

import torch
import numpy as np
from sklearn.base import BaseEstimator

from enchanter.engine import modules


__all__ = [
    "BaseEnsembleEstimator", "SoftEnsemble", "HardEnsemble"
]


class BaseEnsembleEstimator(BaseEstimator):
    def __init__(self, runners, mode=None):
        self.runners = runners
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, x):
        x = modules.numpy2tensor(x).to(self.device)

        predicts = []
        for runner in self.runners:
            predicts.append(runner.predict(x))

        return predicts


class SoftEnsemble(BaseEnsembleEstimator):
    """
    確率の平均をとるアンサンブル
    """

    def predict(self, x):
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
    def __init__(self, runners: List):
        super().__init__(runners, mode="classification")

    def predict(self, x):
        """

        Args:
            x:

        Returns:

        """
        predicts = super().predict(x)
        predicts = np.stack(predicts)

        return np.ravel(predicts[0])
