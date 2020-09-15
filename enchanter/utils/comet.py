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
        各引数のより詳しい説明は https://www.comet.ml/docs/python-sdk/introduction-optimizer/ を参照してください。

        Args:
            algorithm: パラメータチューニングに用いるアルゴリズムを指定します。対応しているアルゴリズムは ['grid', 'random', 'bayes'] です。
            metric: 最小化/最大化する値を指定します。デフォルトでは、'validate_avg_loss' が指定されています。
            objective: metrics を最大化/最小化するかを指定します。['minimize', 'maximize'] で指定してください。
            seed: シード値を設定します。デフォルトでは指定されていません。
            max_combo: 整数、試行するパラメーターの組み合わせの制限（デフォルトは0、制限なしを意味します）
            grid_size: 整数、グリッドを作成するときのパラメーターあたりのビンの数（デフォルトは10）
            min_sample_size: 整数、適切なグリッド範囲を見つけるのに役立つサンプル数（デフォルトは100）
            retry_limit: 整数、中断する前に一意のパラメータセットを作成しようとする制限（デフォルトは20）
            retry_assign_limit:
            name: 文字列、この検索インスタンスに関連付けるパーソナライズ可能な名前（オプション）
            trials: 試行回数を指定します。
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
        カテゴリカル変数を探索する為のメソッドです。

        Args:
            name: 変数の名前を指定します。
            values: 探索する変数をしていします。comet.mlの仕様上、文字列のリストを与える必要があります。

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
        変数を一様分布からサンプリングし探索する為のメソッドです。

        Args:
            name: 変数の名前
            min_value: 最小値
            max_value: 最大値
            dtype: データ型。指定されない場合、`min_value` と `max_value` の値から自動的に推定されます。

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
        変数を正規分布からサンプリングし探索する為のメソッドです。

        Args:
            name: 変数の名前
            min_value: 最小値
            max_value: 最大値
            mu: 正規分布のμ
            sigma: 正規分布のσ
            dtype: データ型。指定されない場合、`min_value` と `max_value` の値から自動的に推定されます。

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
        変数を対数正規分布からサンプリングし探索する為のメソッドです。

        Args:
            name: 変数の名前
            min_value: 最小値
            max_value: 最大値
            mu: 正規分布のμ
            sigma: 正規分布のσ
            dtype: データ型。指定されない場合、`min_value` と `max_value` の値から自動的に推定されます。

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
        変数を対数一様分布からサンプリングし探索する為のメソッドです。

        Args:
            name: 変数の名前
            min_value: 最小値
            max_value: 最大値
            dtype: データ型。指定されない場合、`min_value` と `max_value` の値から自動的に推定されます。

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
        指定した数値型の変数を探索する為のメソッドです。

        Args:
            name: 変数の名前
            values: 数値型の要素で構成されたリスト

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
        `comet_ml.Optimizer` 用のConfigを生成する為のメソッドです。

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
        作成したConfigをファイルにして保存する為のメソッドです。

        """
        with open(filename, "w") as f:
            f.write(self.__repr__())

    def __repr__(self):
        return pformat(self.generate(), indent=1)
