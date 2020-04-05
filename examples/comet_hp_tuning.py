from comet_ml import Optimizer
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
import enchanter.wrappers as wrappers
import enchanter.addons as addons
import enchanter.addons.layers as layers


config = {
    "algorithm": "bayes",

    "parameters": {
        "activation": {"type": "categorical", "values": ["addons.mish", "torch.relu", "torch.sigmoid"]},
    },

    "spec": {
    "metric": "train_avg_loss",
        "objective": "minimize",
    },
    "trials": 5,
}

opt = Optimizer(config)

for experiment in opt.get_experiments():
    model = layers.MLP([4, 512, 128, 3], eval(experiment.get_parameter("activation")))
    optimizer = optim.Adam(model.parameters())
    runner = wrappers.ClassificationRunner(
        model, optimizer=optimizer, criterion=nn.CrossEntropyLoss(), experiment=experiment
    )
    x, y = load_iris(return_X_y=True)
    x = x.astype("float32")
    y = y.astype("int64")

    runner.fit(x, y, epochs=10)
