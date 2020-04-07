# ***************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# ***************************************************


import json

try:
    import comet_ml
except ImportError:
    raise ImportError("You have to install `comet_ml` if you want to use `enchanter.utils.comet` module.")


class TunerConfigGenerator:
    def __init__(
            self, algorithm="bayes", metrics="validate_avg_loss", objective="minimize", seed=None, max_combo=0,
            grid_size=10, min_sample_size=100, retry_limit=20,  retry_assign_limit=0, name=None, trials=1
    ):
        """

        Args:
            algorithm: パラメータチューニングに用いるアルゴリズムを指定します。対応しているアルゴリズムは ['grid', 'random', 'bayes'] です。
            metrics: 最小化/最大化する値を指定します。デフォルトでは、'validate_avg_loss' が指定されています。
            objective: metrics を最大化/最小化するかを指定します。['minimize', 'maximize'] で指定してください。
            seed: シード値を設定します。デフォルトでは指定されていません。
            max_combo:
            grid_size:
            min_sample_size:
            retry_limit:
            retry_assign_limit:
            name:
            trials: 試行回数を指定します。
        """

        self.algorithm = algorithm
        self.spec = {
            "maxCombo": max_combo,
            "objective": objective,
            "metrics": metrics,
            "seed": seed,
            "gridSize": grid_size,
            "minSampleSize": min_sample_size,
            "retryLimit": retry_limit,
            "retryAssignLimit": retry_assign_limit
        }
        self.name = name
        self.trials = trials
        self.__params = {}

    def suggest_categorical(self, name, values):
        """
        カテゴリカル変数を探索する為のメソッドです。

        Args:
            name: 変数の名前を指定します。
            values: 探索する変数をしていします。comet.mlの仕様上、文字列のリストを与える必要があります。

        Examples:
            >>> config = TunerConfigGenerator()
            >>> config.suggest_categorical("activation", ["torch.relu", "torch.sigmoid", "torch.softmax"])
            >>> opt = comet_ml.Optimizer(config.generate())
            >>> for experiment in opt.get_experiments():
            >>>     activation = experiment.get_parameter(eval("activation"))

        """
        self.__params["{}".format(name)] = {
            "type": "categorical",
            "values": [str(value) for value in values]
        }
        return self

    def __suggest(self, name, min_value, max_value, dtype, scaling, **kwargs):
        if dtype is None:
            if type(min_value) is type(max_value):
                dtype = type(min_value)
            else:
                raise Exception("`min_value` and `max_value` must be same data type.")

        if dtype is float:
            dtype = "float"
        elif dtype is int:
            dtype = "integer"
        else:
            raise Exception("`dtype` must be specified as `int` or `float` .")

        self.__params["{}".format(name)] = {
            "type": dtype,
            "min": min_value,
            "max": max_value,
            "scalingType": scaling
        }

        for key in ["mu", "sigma"]:
            if key in kwargs:
                self.__params["{}".format(name)][key] = kwargs[key]

    def suggest_linear(self, name, min_value, max_value, dtype=None):
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

    def suggest_uniform(self, name, min_value, max_value, dtype=None):
        """
        変数を一様分布からサンプリングし探索する為のメソッドです。

        Args:
            name: 変数の名前
            min_value: 最小値
            max_value: 最大値
            dtype: データ型。指定されない場合、'min_value' と 'max_value' の値から自動的に推定されます。

        Examples:
            >>> config = TunerConfigGenerator()
            >>> config.suggest_uniform("uniform", 0.0, 1.0)
            >>> opt = comet_ml.Optimizer(config.generate())
            >>> for experiment in opt.get_experiments():
            >>>     uniform = experiment.get_parameter("uniform")

        """
        self.__suggest(name, min_value, max_value, dtype, "uniform")
        return self

    def suggest_normal(self, name, min_value, max_value, mu=0.0, sigma=1.0, dtype=None):
        """
        変数を正規分布からサンプリングし探索する為のメソッドです。

        Args:
            name: 変数の名前
            min_value: 最小値
            max_value: 最大値
            mu: 正規分布のμ
            sigma: 正規分布のσ
            dtype: データ型。指定されない場合、'min_value' と 'max_value' の値から自動的に推定されます。

        Examples:
            >>> config = TunerConfigGenerator()
            >>> config.suggest_normal("normal", 0.0, 1.0)
            >>> opt = comet_ml.Optimizer(config.generate())
            >>> for experiment in opt.get_experiments():
            >>>     normal = experiment.get_parameter("normal")

        """
        self.__suggest(name, min_value, max_value, dtype, "normal", mu=mu, sigma=sigma)
        return self

    def suggest_lognormal(self, name, min_value, max_value, mu=0.0, sigma=1.0, dtype=None):
        """
        変数を対数正規分布からサンプリングし探索する為のメソッドです。

        Args:
            name: 変数の名前
            min_value: 最小値
            max_value: 最大値
            mu: 正規分布のμ
            sigma: 正規分布のσ
            dtype: データ型。指定されない場合、'min_value' と 'max_value' の値から自動的に推定されます。

        Examples:
            >>> config = TunerConfigGenerator()
            >>> config.suggest_lognormal("lognormal", 0.0, 1.0)
            >>> opt = comet_ml.Optimizer(config.generate())
            >>> for experiment in opt.get_experiments():
            >>>     lognormal = experiment.get_parameter("lognormal")

        """
        self.__suggest(name, min_value, max_value, dtype, "lognormal", mu=mu, sigma=sigma)
        return self

    def suggest_loguniform(self, name, min_value, max_value, dtype=None):
        """
        変数を対数一様分布からサンプリングし探索する為のメソッドです。

        Args:
            name: 変数の名前
            min_value: 最小値
            max_value: 最大値
            dtype: データ型。指定されない場合、'min_value' と 'max_value' の値から自動的に推定されます。

        Examples:
            >>> config = TunerConfigGenerator()
            >>> config.suggest_loguniform("loguniform", 0.0, 1.0)
            >>> opt = comet_ml.Optimizer(config.generate())
            >>> for experiment in opt.get_experiments():
            >>>     loguniform = experiment.get_parameter("loguniform")

        """
        self.__suggest(name, min_value, max_value, dtype, "loguniform")
        return self

    def suggest_discrete(self, name, values):
        """
        指定した数値型の変数を探索する為のメソッドです。

        Args:
            name: 変数の名前
            values: 数値型の要素で構成されたリスト

        Examples:
            >>> config = TunerConfigGenerator()
            >>> config.suggest_discrete("discrete", [10, 20, 30])
            >>> opt = comet_ml.Optimizer(config.generate())
            >>> for experiment in opt.get_experiments():
            >>>     discrete = experiment.get_parameter("discrete")

        """
        self.__params["{}".format(name)] = {
            "type": "discrete",
            "values": values
        }
        return self

    def generate(self):
        """
        'comet_ml.Optimizer' 用のConfigを生成する為のメソッドです。

        Examples:
            >>> config = TunerConfigGenerator()
            >>> config.suggest_discrete("discrete", [10, 20, 30])
            >>> cfg = config.generate()
        """

        config = {
            "algorithm": self.algorithm,
            "parameters": self.__params,
            "spec": self.spec,
            "trials": self.trials,
            "name": self.name
        }
        return json.dumps(config)

    def to_json(self, filename):
        """
        作成したConfigをjson形式で保存する為のメソッドです。

        """
        with open(filename, "w") as f:
            json.dump(self.generate(), f, indent=2, ensure_ascii=False)
