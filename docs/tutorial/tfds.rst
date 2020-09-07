Training PyTorch model using TensorFlow Dataset
================================================

PyTorch標準の ``torch.utils.data.Dataset`` と ``torch.utils.data.DataLoader`` は非常に使いやすいAPIを持つ一方、
実行速度が遅いという問題があります。

一方、TensorFlow Dataset APIは非常に実行速度が速いのが特徴です。
Enchanterでは、PyTorch DataLoaderに加えTensorFlow Datasetを試験的にサポートしています。
早速、簡単な例を書いてみましょう。

TensorFlow Dataset自体にも Irisデータセットを読み込むためのクラスが用意されていますが、
ここではあえて ``sklearn.datasets`` を用います。

::

    from comet_ml import Experiment

    import numpy as np
    import tensorflow as tf
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    import enchanter.tasks as tasks
    import enchanter.addons as addons

    x, y = load_iris(return_X_y=True)
    x = x.astype(np.float32)
    y = y.astype(np.int64)

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=0)

    batch_size = 32


これで必要なモジュールのインポートとデータセットの準備ができました。
次に ``tf.data.Dataset.from_tensor_slices`` を使って ``Dataset`` クラスを作成し、シャッフル操作とバッチ数を指定します。

::

    train_loader = tf.data.Dataset.from_tensor_slices(
                        (x_train, y_train)
                    ).shuffle(batch_size).batch(batch_size)

    val_loader = tf.data.Dataset.from_tensor_slices(
                        (x_val, y_val)
                    ).shuffle(batch_size).batch(batch_size)

    test_loader = tf.data.Dataset.from_tensor_slices(
                        (x_test, y_test)
                    ).shuffle(batch_size).batch(batch_size)

これで準備が完了しました。次に適当なモデルを用意し、学習させましょう。

::

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
