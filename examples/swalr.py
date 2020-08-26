from comet_ml import Experiment
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from enchanter.tasks import ClassificationRunner


import models


train_ds = MNIST(
    "../tests/data",
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

test_ds = MNIST(
    "../tests/data",
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)


train_loader = DataLoader(train_ds, batch_size=32)
test_loader = DataLoader(test_ds, batch_size=32)


class SWALRRunner(ClassificationRunner):
    def __init__(self, *args, **kwargs):
        super(SWALRRunner, self).__init__(*args, **kwargs)
        self.swa_model = AveragedModel(self.model)
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=0.05)
        self.swa_start = 5

    def update_scheduler(self, epoch: int) -> None:
        if epoch > self.swa_start:
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()

        else:
            super(SWALRRunner, self).update_scheduler(epoch)

    def train_end(self, outputs):
        update_bn(self.loaders["train"], self.swa_model)
        return super(SWALRRunner, self).train_end(outputs)


model = models.MNIST()
optimizer = optim.Adam(model.parameters())

runner = SWALRRunner(
    model, optimizer, nn.CrossEntropyLoss(),
    experiment=Experiment(), scheduler=[ExponentialLR(optimizer, gamma=0.9)]
)
runner.add_loader("train", train_loader)
runner.add_loader("test", test_loader)
runner.train_config(epochs=10)
runner.run()
