from .classification import ClassificationRunner
from .regression import RegressionRunner
from .ensemble import HardEnsemble, SoftEnsemble

__all__ = [
    "ClassificationRunner", "RegressionRunner",
    "HardEnsemble", "SoftEnsemble"
]


try:
    from enchanter.wrappers.comet import ConfigGenerator
except ImportError:
    pass
else:
    __all__.append("ConfigGenerator")
