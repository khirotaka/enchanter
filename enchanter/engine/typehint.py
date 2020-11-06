from typing import Union
from sklearn.linear_model import LinearRegression
from sklearn.tree import BaseDecisionTree
from sklearn.svm._base import BaseLibSVM
from sklearn.neighbors._base import NeighborsBase

__all__ = ["ScikitModel"]


ScikitModel = Union[Union[LinearRegression, BaseDecisionTree], Union[BaseLibSVM, NeighborsBase]]
