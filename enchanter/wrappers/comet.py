import json

try:
    import comet_ml
except ImportError:
    raise ImportError("You have to install `comet_ml` if you want to use `enchanter.wrappers.comet` module.")


class ConfigGenerator:
    def __init__(
            self, algorithm="bayes", metrics="validate_avg_loss", objective="minimize", seed=None, max_combo=0,
            grid_size=10, min_sample_size=100,  retry_limit=20,  retry_assign_limit=0, name=None, trials=1
    ):
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
        self.__params["{}".format(name)] = {
            "type": "categorical",
            "values": [str(value) for value in values]
        }

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
        self.__suggest(name, min_value, max_value, dtype, "linear")

    def suggest_uniform(self, name, min_value, max_value, dtype=None):
        self.__suggest(name, min_value, max_value, dtype, "uniform")

    def suggest_normal(self, name, min_value, max_value, mu=0.0, sigma=1.0, dtype=None):
        self.__suggest(name, min_value, max_value, dtype, "normal", mu=mu, sigma=sigma)

    def suggest_lognormal(self, name, min_value, max_value, mu=0.0, sigma=1.0, dtype=None):
        self.__suggest(name, min_value, max_value, dtype, "lognormal", mu=mu, sigma=sigma)

    def suggest_loguniform(self, name, min_value, max_value, dtype=None):
        self.__suggest(name, min_value, max_value, dtype, "loguniform")

    def suggest_discrete(self, name, values):
        self.__params["{}".format(name)] = {
            "type": "discrete",
            "values": values
        }

    def generate(self):
        config = {
            "algorithm": self.algorithm,
            "parameters": self.__params,
            "spec": self.spec,
            "trials": self.trials,
            "name": self.name
        }
        return config

    def to_json(self):
        return json.dumps(self.generate())
