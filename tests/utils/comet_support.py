try:
    from comet_ml import Optimizer
    from comet_ml import exceptions

    _COMET_AVAILABLE = True

except ImportError:
    _COMET_AVAILABLE = False

try:
    from enchanter.utils.comet import TunerConfigGenerator

    _NOT_INSTALL_COMET = False

except ImportError:
    _NOT_INSTALL_COMET = True


def test_import_utils():
    if not _COMET_AVAILABLE:
        assert _NOT_INSTALL_COMET
    else:
        assert True


def test_config_gen_1():
    if _COMET_AVAILABLE:
        is_pass = False
        try:
            config = TunerConfigGenerator(algorithm="foo")
        except ValueError:
            is_pass = True

        except Exception as e:
            print(e)

        assert is_pass
    else:
        assert True


def test_config_gen_2():
    if _COMET_AVAILABLE:
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
    else:
        assert True
