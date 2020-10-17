Unsupervised Time Series Runner
====================================


論文 `Unsupervised Scalable Representation Learning for Multivariate Time Series <https://papers.nips.cc/paper/8713-unsupervised-scalable-representation-learning-for-multivariate-time-series>`_
で提案された時系列用教師なし表現学習法が利用可能になりました。
ここでは、実際の時系列データセットを利用して教師なしで特徴抽出器を作成する方法を学びます。

..  warning::

    現在、入力サンプルの長さが一定でない方式をサポートしていません。将来のリリースで対応予定です。


``tslearn`` を用いてUCR/UEAアーカイブに収録されている ``Libras`` データセット用の特徴抽出器を作成します。
なお、詳細なアルゴリズムについては、元の論文を参照してください。

まず、必要なライブラリをインポートし、データセットを準備します。

::

    from comet_ml import Experiment

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


次に任意のニューラルネットを用意します。ここでは、``Causal Convolution`` を要素に持った ``ResBlock`` である、
``Temporal Convolution Block`` を利用したモデルを用います。


::

    class Encoder(nn.Module):
        def __init__(self, in_features, mid_features, out_features, representation_size):
            super(Encoder, self).__init__()
            self.conv = nn.Sequential(
                L.TemporalConvBlock(in_features, mid_features, 3, dilation=2**0, activation=nn.LeakyReLU()),
                L.TemporalConvBlock(mid_features, mid_features, 3, dilation=2**1, activation=nn.LeakyReLU()),
                L.TemporalConvBlock(mid_features, mid_features, 3, dilation=2**2, activation=nn.LeakyReLU()),
                L.TemporalConvBlock(mid_features, mid_features, 3, dilation=2**3, activation=nn.LeakyReLU()),
                L.TemporalConvBlock(mid_features, mid_features, 3, dilation=2**4, activation=nn.LeakyReLU()),
                L.TemporalConvBlock(mid_features, mid_features, 3, dilation=2**5, activation=nn.LeakyReLU()),
                L.TemporalConvBlock(mid_features, mid_features, 3, dilation=2**6, activation=nn.LeakyReLU()),
                L.TemporalConvBlock(mid_features, mid_features, 3, dilation=2**7, activation=nn.LeakyReLU()),
                L.TemporalConvBlock(mid_features, mid_features, 3, dilation=2**8, activation=nn.LeakyReLU()),
                L.TemporalConvBlock(mid_features, out_features, 3, dilation=2**9, activation=nn.LeakyReLU()),
                nn.AdaptiveMaxPool1d(1),
            )
            self.fc = nn.Linear(out_features, representation_size)

        def forward(self, x):
            batch = x.shape[0]
            out = self.conv(x).reshape(batch, -1)
            return self.fc(out)


次に、DataLoader周りを実装します。

::

    train_ds = TimeSeriesLabeledDataset(x_train, y_train)
    test_ds = TimeSeriesLabeledDataset(x_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)


これで準備完了です。最後の学習コードを書きましょう。

::

    experiment = Experiment()
    model = Encoder(x_train.shape[1], 40, 160, 320)
    optimizer = optim.Adam(model.parameters())

    runner = TimeSeriesUnsupervisedRunner(
        model, optimizer, experiment, 10, 1,
    )

    runner.train_config(epochs=100)
    runner.add_loader("train", train_loader)
    runner.add_loader("val", test_loader)
    runner.add_loader("test", test_loader)
    runner.run()


なお、このタスク専用に作成されたEarly Stoppingツール ``enchanter.callbacks.EarlyStoppingForTSUS`` も利用可能です。
これは、以下の様にして用います。

::

    runner = TimeSeriesUnsupervisedRunner(
        model, optimizer, experiment, 10, 1,
        callbacks=[EarlyStoppingForTSUS(x_train, y_train)],
    )


このモジュールにはモデルの汎化性能を評価するために``sklearm.svm.SVC`` が内蔵されているため、それを訓練・評価するためにデータとラベルのペアが必要です。
また、グリッドサーチでSVMに最適なハイパーパラメータを探索することも可能です。

::

    grid_search = {
        "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, np.inf],
        "kernel": ["rbf"],
        "degree": [3],
        "gamma": ["scale"],
        "coef0": [0],
        "shrinking": [True],
        "probability": [False],
        "tol": [0.001],
        "cache_size": [200],
        "class_weight": [None],
        "verbose": [False],
        "max_iter": [10000000],
        "decision_function_shape": ["ovr"],
        "random_state": [None],
    }

    runner = TimeSeriesUnsupervisedRunner(
        model, optimizer, experiment, 10, 1,
        callbacks=[EarlyStoppingForTSUS(x_train, y_train, grid_search=grid_search))]
    )
