from comet_ml import Experiment

import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

import enchanter.tasks as tasks
import enchanter.addons as addons
import enchanter.addons.layers as layers


experiment = Experiment()
model = layers.MLP([4, 512, 128, 3], addons.mish)
optimizer = optim.Adam(model.parameters())
runner = tasks.ClassificationRunner(
    model, optimizer=optimizer, criterion=nn.CrossEntropyLoss(), experiment=experiment
)
x, y = load_iris(return_X_y=True)
x = x.astype("float32")
y = y.astype("int64")

runner.fit(x, y, epochs=10)
predict = runner.predict(x)
accuracy = accuracy_score(y, predict)
experiment.log_metric("test_accuracy", accuracy)
