from comet_ml import Experiment
# from enchanter.callbacks import TensorBoardLogger as Experiment
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tslearn.datasets import UCR_UEA_datasets
from enchanter.addons import layers as L
from enchanter.callbacks import EarlyStoppingForTSUS
from enchanter.tasks import TimeSeriesUnsupervisedRunner
from enchanter.engine.modules import fix_seed
from enchanter.utils.datasets import TimeSeriesLabeledDataset


fix_seed(800)

downloader = UCR_UEA_datasets()
x_train, y_train, x_test, y_test = downloader.load_dataset("Libras")
x_train = torch.tensor(x_train.transpose(0, 2, 1), dtype=torch.float32)
x_test = torch.tensor(x_test.transpose(0, 2, 1), dtype=torch.float32)

y_train = y_train.astype(float).astype(int) - 1
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = y_test.astype(float).astype(int) - 1
y_test = torch.tensor(y_test, dtype=torch.long)


class Encoder(nn.Module):
    def __init__(self, in_features, mid_features, out_features):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            L.CausalConv1d(in_features, mid_features, 3),
            nn.LeakyReLU(),
            L.CausalConv1d(mid_features, mid_features, 3),
            nn.LeakyReLU(),
            L.CausalConv1d(mid_features, mid_features, 3),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.fc = nn.Linear(mid_features, out_features)

    def forward(self, x):
        batch = x.shape[0]
        out = self.conv(x).reshape(batch, -1)
        return self.fc(out)


experiment = Experiment()
experiment.add_tag("ts-us")

train_ds = TimeSeriesLabeledDataset(x_train, y_train)
test_ds = TimeSeriesLabeledDataset(x_test, y_test)

train_loader = DataLoader(train_ds, batch_size=32)
test_loader = DataLoader(test_ds, batch_size=32)

model = Encoder(x_train.shape[1], 30, 100)
optimizer = optim.Adam(model.parameters())

runner = TimeSeriesUnsupervisedRunner(
    model, optimizer, experiment, 10, 1,
    callbacks=[EarlyStoppingForTSUS(x_train, y_train)]
)

runner.train_config(10)
runner.add_loader("train", train_loader)
runner.add_loader("val", test_loader)
runner.run()
