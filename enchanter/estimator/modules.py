import comet_ml
from typing import Dict


def is_jupyter() -> bool:
    if "get_ipython" not in globals():
        return False
    env = get_ipython().__class__.__name__
    if env == "TerminalInteractiveShell":
        return False
    return True


class CometLogger:
    def __init__(self, experiment):
        self.experiment: comet_ml.Experiment = experiment

    def log_train(self, epoch: int, step: int, values: Dict):
        with self.experiment.train():
            for k in values.keys():
                self.experiment.log_metric(k, values[k], step=step, epoch=epoch)

    def log_val(self, epoch: int, step: int, values: Dict):
        with self.experiment.validate():
            for k in values.keys():
                self.experiment.log_metric(k, values[k], step=step, epoch=epoch)

    def log_test(self, values: Dict):
        with self.experiment.test():
            for k in values.keys():
                self.experiment.log_metric(k, values[k])
