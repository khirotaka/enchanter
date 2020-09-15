# *******************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# *******************************************************

from typing import List, Union, Optional

from numpy import stack, ravel, ndarray, int as np_int
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

    def predict(self, x: Union[ndarray, Tensor]) -> List[ndarray]:
        x = as_tensor(x, device=self.device)

        predicts = []
        for runner in self.runners:
            predicts.append(runner.predict(x))

        return predicts


class SoftEnsemble(BaseEnsembleEstimator):
    """
    確率の平均をとるアンサンブル
    """

    def predict(self, x: Union[ndarray, Tensor]) -> ndarray:
        """

        Args:
            x:

        Returns:

        """
        predicts = super(SoftEnsemble, self).predict(x)

        predicts = sum(predicts)
        probs = predicts / len(self.runners)

        if self.mode == "classification":
            probs = probs.astype(np_int)

        return probs


class HardEnsemble(BaseEnsembleEstimator):
    """
    多数決をとるアンサンブル
    """

    def __init__(self, runners: List[BaseRunner]) -> None:
        super(HardEnsemble, self).__init__(runners, mode="classification")

    def predict(self, x: Union[ndarray, Tensor]) -> ndarray:
        """

        Args:
            x:

        Returns:

        """
        predicts = super().predict(x)
        predicts = stack(predicts)

        return ravel(predicts[0])
