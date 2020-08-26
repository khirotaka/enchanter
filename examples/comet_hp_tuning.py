from comet_ml import Optimizer
import torch                            # pylint: disable=W0611
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
import enchanter.tasks as tasks
import enchanter.addons as addons       # pylint: disable=W0611
import enchanter.addons.layers as layers
from enchanter.utils import comet

config = comet.TunerConfigGenerator(
    algorithm="bayes",
    metric="train_avg_loss",
    objective="minimize",
    seed=0,
    trials=5
)

config.suggest_categorical("activation", ["addons.mish", "torch.relu", "torch.sigmoid"])

opt = Optimizer(config.generate())

for experiment in opt.get_experiments():
    model = layers.MLP([4, 512, 128, 3], eval(experiment.get_parameter("activation")))
    optimizer = optim.Adam(model.parameters())
    runner = tasks.ClassificationRunner(
        model, optimizer=optimizer, criterion=nn.CrossEntropyLoss(), experiment=experiment
    )
    x, y = load_iris(return_X_y=True)
    x = x.astype("float32")
    y = y.astype("int64")

    runner.fit(x, y, epochs=1, batch_size=32)
