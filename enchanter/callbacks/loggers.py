from contextlib import contextmanager
from torch.utils.tensorboard import SummaryWriter


class BaseLogger:
    def __init__(self):
        self.context = None

    @contextmanager
    def train(self):
        old_state = self.context
        self.context = "train"

        yield self

        self.context = old_state

    @contextmanager
    def validate(self):
        old_state = self.context
        self.context = "validate"

        yield self

        self.context = old_state

    @contextmanager
    def test(self):
        old_state = self.context
        self.context = "test"

        yield self

        self.context = old_state

    def log_metric(self, name, value, step=None, epoch=None, include_context=True):
        pass

    def log_metrics(self, dic, prefix=None, step=None, epoch=None):
        pass

    def log_parameter(self, name, value, step=None):
        pass

    def log_parameters(self, dic, prefix=None, step=None):
        pass

    def set_model_graph(self, *args, **kwargs):
        pass


class TensorBoardLogger(BaseLogger):
    def __init__(self, log_dir, *args, **kwargs):
        super().__init__()
        self.writer = SummaryWriter(log_dir, *args, **kwargs)

    def log_metric(self, name, value, step=None, epoch=None, include_context=True):
        self.writer.add_scalar("{}/{}".format(self.context, name), value)

    def log_metrics(self, dic, prefix=None, step=None, epoch=None):
        for k, v in dic.items():
            self.log_metric(k, v, step, epoch)

    def log_parameter(self, name, value, step=None):
        self.writer.add_scalar("{}/{}/{}".format(self.context, "params", name), value, step)

    def log_parameters(self, dic: dict, prefix=None, step=None):
        for k, v in dic.items():
            self.log_parameter(k, v)
