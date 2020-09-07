from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.datasets import load_iris
from tqdm.auto import tqdm

import enchanter.tasks as tasks
import enchanter.engine.modules as modules
import enchanter.addons as addons
import enchanter.addons.layers as layers


experiment = Experiment()
model = layers.MLP([4, 512, 128, 3], addons.mish)
optimizer = optim.Adam(model.parameters())

x, y = load_iris(return_X_y=True)
x = x.astype("float32")
y = y.astype("int64")

train_ds = modules.get_dataset(x, y)
val_ds = modules.get_dataset(x, y)
test_ds = modules.get_dataset(x, y)

train_loader = DataLoader(train_ds, batch_size=32)
val_loader = DataLoader(val_ds, batch_size=32)
test_loader = DataLoader(test_ds, batch_size=32)


with tasks.ClassificationRunner(model, optimizer, nn.CrossEntropyLoss(), experiment) as runner:
    for epoch in tqdm(range(10)):
        with runner.experiment.train():
            for train_batch in train_loader:
                runner.optimizer.zero_grad()
                train_out = runner.train_step(train_batch)
                runner.backward(train_out["loss"])
                runner.update_optimizer()

                with runner.experiment.validate(), torch.no_grad():
                    for val_batch in val_loader:
                        val_out = runner.val_step(val_batch)["loss"]
                        runner.experiment.log_metric("val_loss", val_out)

        with runner.experiment.test(), torch.no_grad():
            for test_batch in test_loader:
                test_out = runner.test_step(test_batch)["loss"]
                runner.experiment.log_metric("test_loss", test_out)

runner.save("checkpoint")
