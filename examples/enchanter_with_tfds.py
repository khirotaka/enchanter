from comet_ml import Experiment

import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import tensorflow as tf

import enchanter.tasks as tasks
import enchanter.addons as addons

x, y = load_iris(return_X_y=True)
x = x.astype(np.float32)
y = y.astype(np.int64)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=0)

batch_size = 32

train_loader = \
    tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)
    ).shuffle(batch_size).batch(batch_size)

val_loader = \
    tf.data.Dataset.from_tensor_slices(
        (x_val, y_val)
    ).shuffle(batch_size).batch(batch_size)
test_loader = \
    tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)
    ).shuffle(batch_size).batch(batch_size)

model = addons.layers.MLP([4, 16, 3], activation=addons.Mish())
optimizer = optim.Adam(model.parameters())

runner = tasks.ClassificationRunner(
    model,
    optimizer,
    nn.CrossEntropyLoss(),
    Experiment()
)
runner.add_loader("train", train_loader).add_loader("val", val_loader).add_loader("test", test_loader)
runner.train_config(epochs=10)
runner.run()
