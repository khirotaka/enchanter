from comet_ml import Optimizer
from comet_ml import exceptions
from enchanter.utils.comet import TunerConfigGenerator


def test_config_gen_1():
    is_pass = False
    try:
        config = TunerConfigGenerator(algorithm="foo")
    except ValueError:
        is_pass = True

    except Exception as e:
        print(e)

    assert is_pass


def test_config_gen_2():
    config = TunerConfigGenerator()
    config.suggest_categorical("categorical", ["a", "b", "c"])
    config.suggest_uniform("uniform", 0.0, 1.0)
    config.suggest_normal("normal", 0.0, 1.0)
    config.suggest_lognormal("lognormal", 0.0, 1.0)
    config.suggest_loguniform("loguniform", 0.0, 1.0)
    config.suggest_discrete("discrete", [10, 20, 30])

    config.export("test.config")

    is_pass = False
    try:
        opt = Optimizer("test.config", experiment_class="OfflineExperiment")

    except exceptions.OptimizerException as e:
        print(e)
        is_pass = False
    except Exception as e:
        print(e)

    else:
        is_pass = True
    assert is_pass
