# *******************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# *******************************************************

from typing import List, Union, Optional

import numpy as np
from sklearn.base import BaseEstimator
from torch import as_tensor, device, Tensor
from torch.cuda import is_available
from enchanter.engine.runner import BaseRunner

__all__ = ["BaseEnsembleEstimator", "SoftEnsemble", "HardEnsemble"]


class BaseEnsembleEstimator(BaseEstimator):
    def __init__(self, runners: List[BaseRunner], mode: Optional[str] = None) -> None:
        self.runners: List[BaseRunner] = runners
        self.mode: Optional[str] = mode
        self.device: device = device("cuda" if is_available() else "cpu")

    def predict(self, x: Union[np.ndarray, Tensor]) -> List[np.ndarray]:
        x = as_tensor(x, device=self.device)

        predicts = []
        for runner in self.runners:
            predicts.append(runner.predict(x))

        return predicts


class SoftEnsemble(BaseEnsembleEstimator):
    """
    Ensemble that averages probabilities
    """

    def predict(self, x: Union[np.ndarray, Tensor]) -> np.ndarray:
        predicts = super(SoftEnsemble, self).predict(x)

        predicts = sum(predicts)
        probs = predicts / len(self.runners)

        if self.mode == "classification":
            probs = probs.astype(np.int)

        return probs


class HardEnsemble(BaseEnsembleEstimator):
    """
    Ensemble that takes a majority vote
    """

    def __init__(self, runners: List[BaseRunner]) -> None:
        super(HardEnsemble, self).__init__(runners, mode="classification")

    def predict(self, x: Union[np.ndarray, Tensor]) -> np.ndarray:
        predicts = super().predict(x)
        predicts = np.stack(predicts)

        return np.ravel(predicts[0])
