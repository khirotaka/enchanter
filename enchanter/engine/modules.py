# *******************************************************
#  _____            _                 _
# | ____|_ __   ___| |__   __ _ _ __ | |_ ___ _ __
# |  _| | '_ \ / __| '_ \ / _` | '_ \| __/ _ \ '__|
# | |___| | | | (__| | | | (_| | | | | ||  __/ |
# |_____|_| |_|\___|_| |_|\__,_|_| |_|\__\___|_|
#
# *******************************************************

from typing import Dict

import torch
import numpy as np
from torch.utils.data import TensorDataset, Dataset


def is_jupyter() -> bool:
    if "get_ipython" not in globals():
        return False
    env = get_ipython().__class__.__name__
    if env == "TerminalInteractiveShell":
        return False
    return True


def numpy2tensor(inputs: np.ndarray) -> torch.Tensor:
    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs)

    return inputs


def get_dataset(x, y=None) -> Dataset:
    x = numpy2tensor(x)

    if y is not None:
        y = numpy2tensor(y)
        ds = TensorDataset(x, y)
    else:
        ds = TensorDataset(x)

    return ds


class CometLogger:
    def __init__(self, experiment):
        self.experiment = experiment

    def log_histogram_3d(self, values, name, step):
        self.experiment.log_histogram_3d(values, name, step)

    def add_tag(self, tag):
        self.experiment.add_tag(tag)

    def log_params(self, dic, prefix=None, step=None):
        self.experiment.log_parameters(dic, prefix, step)

    def log_confusion_matrix(
            self, y_true=None, y_predicted=None, matrix=None,
            labels=None, title="Confusion Matrix", row_label="Actual Category", column_label="Predicted Category",
            max_examples_per_cell=25, max_categories=25, winner_function=None, index_to_example_function=None,
            cache=True, file_name="confusion-matrix.json", overwrite=False, step=None, **kwargs
    ):
        self.experiment.log_confusion_matrix(
            y_true=y_true,
            y_predicted=y_predicted,
            matrix=matrix,
            labels=labels,
            title=title,
            row_label=row_label,
            column_label=column_label,
            max_examples_per_cell=max_examples_per_cell,
            max_categories=max_categories,
            winner_function=winner_function,
            index_to_example_function=index_to_example_function,
            cache=cache,
            file_name=file_name,
            overwrite=overwrite,
            step=step, **kwargs
        )

    def log_graph(self, graph):
        self.experiment.set_model_graph(graph)

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
