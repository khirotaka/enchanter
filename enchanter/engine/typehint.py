from typing_extensions import Protocol


__all__ = ["ScikitModel"]


class ScikitModel(Protocol):
    # pylint: disable=R0201
    def fit(self, X, y, sample_weight=None):
        # pylint: disable=W0104
        ...

    # pylint: disable=R0201
    def predict(self, X):
        # pylint: disable=W0104
        ...

    # pylint: disable=R0201
    def score(self, X, y, sample_weight=None):
        # pylint: disable=W0104
        ...

    # pylint: disable=R0201
    def set_params(self, **params):
        # pylint: disable=W0104
        ...
