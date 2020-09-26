# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************

from typing import Dict, Optional, List, Any, Union
from pprint import pformat
from pkg_resources import working_set


if "comet-ml" not in [d.project_name for d in working_set]:
    raise ImportError("You have to install `comet_ml` if you want to use `enchanter.utils.comet` module.")


__all__ = ["TunerConfigGenerator"]


class TunerConfigGenerator:
    def __init__(
        self,
        algorithm: str = "bayes",
        metric: str = "validate_avg_loss",
        objective: str = "minimize",
        seed: Optional[int] = None,
        max_combo: int = 0,
        grid_size: int = 10,
        min_sample_size: int = 100,
        retry_limit: int = 20,
        retry_assign_limit: int = 0,
        name: Optional[str] = None,
        trials: int = 1,
    ) -> None:
        """
        See https://www.comet.ml/docs/python-sdk/introduction-optimizer/ for a more detailed explanation of each argument.

        Args:
            algorithm: Specifies the algorithm used for parameter tuning. The supported algorithms are ``['grid','random','bayes']``.
            metric: Specify the value to be minimized / maximized. By default,``validate_avg_loss`` is specified.
            objective: Specifies whether to maximize / minimize metrics. Specify with ``['minimize','maximize']``.
            seed: Set the seed value. Not specified by default.
            max_combo: Integer. Limit on the combination of parameters to try (default is 0, meaning no limit)
            grid_size: Integer. Number of bins per parameter when creating a grid (default is 10).
            min_sample_size: Integer. Number of samples to help find a suitable grid range (default is 100).
            retry_limit: integer. A limit that attempts to create a unique set of parameters before suspending (default is 20).
            retry_assign_limit:
            name: String. A personalizable name associated with this search instance. (option)
            trials: Specifies the number of trials in a single experiment.
        """

        if algorithm not in ["grid", "bayes", "random"]:
            raise ValueError("The algorithms you can select are `random`, `bayes` and `grid`.")

        self.algorithm: str = algorithm
        self.spec: Dict[str, Any] = {
            "maxCombo": max_combo,
            "objective": objective,
            "metric": metric,
            "seed": seed,
            "gridSize": grid_size,
            "minSampleSize": min_sample_size,
            "retryLimit": retry_limit,
            "retryAssignLimit": retry_assign_limit,
        }
        self.name: Optional[str] = name
        self.trials: int = trials
        self.__params: Dict[str, Dict[str, Any]] = {}

    def suggest_categorical(self, name: str, values: List[str]):
        """
        A method for searching categorical variables.

        Args:
            name: Specify the name of the variable.
            values: I have a variable to search. Due to the specifications of comet.ml, it is necessary to give a list of strings.

        Examples:
            >>> import comet_ml
            >>> from enchanter.utils.comet import TunerConfigGenerator
            >>> config = TunerConfigGenerator()
            >>> config.suggest_categorical("activation", ["torch.relu", "torch.sigmoid", "torch.softmax"])
            >>> opt = comet_ml.Optimizer(config.generate())
            >>> for experiment in opt.get_experiments():
            >>>     activation = experiment.get_parameter(eval("activation"))

        """
        self.__params["{}".format(name)] = {
            "type": "categorical",
            "values": [str(value) for value in values],
        }
        return self

    def __suggest(
        self,
        name: str,
        min_value: Union[float, int],
        max_value: Union[float, int],
        dtype: Optional[type],
        scaling: str,
        **kwargs,
    ) -> None:
        if dtype is None:
            if type(min_value) is type(max_value):
                dtype = type(min_value)
            else:
                raise Exception("`min_value` and `max_value` must be same data type.")

        if dtype is float:
            dtype_str = "float"
        elif dtype is int:
            dtype_str = "integer"
        else:
            raise Exception("`dtype` must be specified as `int` or `float` .")

        self.__params["{}".format(name)] = {
            "type": dtype_str,
            "min": min_value,
            "max": max_value,
            "scalingType": scaling,
        }

        for key in ["mu", "sigma"]:
            if key in kwargs:
                self.__params["{}".format(name)][key] = kwargs[key]

    def suggest_linear(
        self,
        name: str,
        min_value: Union[float, int],
        max_value: Union[float, int],
        dtype: Optional[type] = None,
    ):
        """
        If Integer, independent distribution. else if float the same as uniform.

        Args:
            name:
            min_value:
            max_value:
            dtype:

        Returns:

        """
        self.__suggest(name, min_value, max_value, dtype, "linear")
        return self

    def suggest_uniform(
        self,
        name: str,
        min_value: Union[float, int],
        max_value: Union[float, int],
        dtype: Optional[type] = None,
    ):
        """
        This is a method for sampling and searching variables from a uniform distribution.

        Args:
            name: Variable name
            min_value: minimum value
            max_value: Maximum value
            dtype: Data type. If not specified, it is automatically estimated from the values of ``min_value`` and` `max_value``.

        Examples:
            >>> import comet_ml
            >>> config = TunerConfigGenerator()
            >>> config.suggest_uniform("uniform", 0.0, 1.0)
            >>> opt = comet_ml.Optimizer(config.generate())
            >>> for experiment in opt.get_experiments():
            >>>     uniform = experiment.get_parameter("uniform")

        """
        self.__suggest(name, min_value, max_value, dtype, "uniform")
        return self

    def suggest_normal(
        self,
        name: str,
        min_value: Union[float, int],
        max_value: Union[float, int],
        mu: float = 0.0,
        sigma: float = 1.0,
        dtype: Optional[type] = None,
    ):
        """
        This is a method for sampling and searching variables from a normal distribution.

        Args:
            name: Variable name
            min_value: minimum value
            max_value: Maximum value
            mu: Normal distribution μ
            sigma: Normal distribution σ
            dtype: Data type. If not specified, it is automatically estimated from the values of ``min_value`` and` `max_value``.

        Examples:
            >>> import comet_ml
            >>> config = TunerConfigGenerator()
            >>> config.suggest_normal("normal", 0.0, 1.0)
            >>> opt = comet_ml.Optimizer(config.generate())
            >>> for experiment in opt.get_experiments():
            >>>     normal = experiment.get_parameter("normal")

        """
        self.__suggest(name, min_value, max_value, dtype, "normal", mu=mu, sigma=sigma)
        return self

    def suggest_lognormal(
        self,
        name: str,
        min_value: Union[float, int],
        max_value: Union[float, int],
        mu: float = 0.0,
        sigma: float = 1.0,
        dtype: Optional[type] = None,
    ):
        """
        A method for sampling and searching variables from a lognormal distribution.

        Args:
            name: Variable name
            min_value: minimum value
            max_value: Maximum value
            mu: Normal distribution μ
            sigma: Normal distribution σ
            dtype: Data type. If not specified, it is automatically estimated from the values of ``min_value`` and` `max_value``.

        Examples:
            >>> import comet_ml
            >>> config = TunerConfigGenerator()
            >>> config.suggest_lognormal("lognormal", 0.0, 1.0)
            >>> opt = comet_ml.Optimizer(config.generate())
            >>> for experiment in opt.get_experiments():
            >>>     lognormal = experiment.get_parameter("lognormal")

        """
        self.__suggest(name, min_value, max_value, dtype, "lognormal", mu=mu, sigma=sigma)
        return self

    def suggest_loguniform(
        self,
        name: str,
        min_value: Union[float, int],
        max_value: Union[float, int],
        dtype: Optional[type] = None,
    ):
        """
        This is a method for sampling and searching variables from a uniform distribution.

        Args:
            name: Variable name
            min_value: minimum value
            max_value: Maximum value
            dtype: Data type. If not specified, it is automatically estimated from the values of ``min_value`` and` `max_value``.

        Examples:
            >>> import comet_ml
            >>> config = TunerConfigGenerator()
            >>> config.suggest_loguniform("loguniform", 0.0, 1.0)
            >>> opt = comet_ml.Optimizer(config.generate())
            >>> for experiment in opt.get_experiments():
            >>>     loguniform = experiment.get_parameter("loguniform")

        """
        self.__suggest(name, min_value, max_value, dtype, "loguniform")
        return self

    def suggest_discrete(self, name: str, values: List[Union[float, int]]):
        """
        This method is used to search for the specified numeric type variable.

        Args:
            name: Variable name
            values: A list of numeric elements

        Examples:
            >>> import comet_ml
            >>> config = TunerConfigGenerator()
            >>> config.suggest_discrete("discrete", [10, 20, 30])
            >>> opt = comet_ml.Optimizer(config.generate())
            >>> for experiment in opt.get_experiments():
            >>>     discrete = experiment.get_parameter("discrete")

        """
        self.__params["{}".format(name)] = {"type": "discrete", "values": values}
        return self

    def generate(self) -> Dict[str, Any]:
        """
        A method for generating a Config for ``comet_ml.Optimizer``.

        Examples:
            >>> cfg = TunerConfigGenerator()
            >>> cfg.suggest_discrete("discrete", [10, 20, 30])
            >>> cfg_dict = cfg.generate()
        """

        config = {
            "algorithm": self.algorithm,
            "parameters": self.__params,
            "spec": self.spec,
            "trials": self.trials,
            "name": self.name,
        }
        return config

    def export(self, filename: str) -> None:
        """
        It is a method to save the created Config as a file.

        """
        with open(filename, "w") as f:
            f.write(self.__repr__())

    def __repr__(self):
        return pformat(self.generate(), indent=1)
