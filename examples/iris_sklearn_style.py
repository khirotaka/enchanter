from comet_ml import Experiment
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
import enchanter.wrappers as wrappers
import enchanter.addons as addons
import enchanter.addons.layers as layers


experiment = Experiment()
model = layers.MLP([4, 512, 128, 3], addons.mish)
optimizer = optim.Adam(model.parameters())
runner = wrappers.ClassificationRunner(
    model, optimizer=optimizer, criterion=nn.CrossEntropyLoss(), experiment=experiment
)
x, y = load_iris(return_X_y=True)
x = x.astype("float32")
y = y.astype("int64")

runner.fit(x, y, epochs=10)
