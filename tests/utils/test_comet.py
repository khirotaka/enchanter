from comet_ml import Optimizer
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
