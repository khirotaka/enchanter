Stochastic Weight Averaging with Enchanter
=================================================

PyTorch v1.6 から Stochastic Weight Averaging (SWA)を行うクラスが導入されました。
このチュートリアルでは、SWAを行うためのRunnerの定義について紹介します。

データセットはMNISTを使います。データセットのダウンロードとMNIST用のCNNを実装しまうす。

::

    from comet_ml import Experiment

    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import ExponentialLR
    from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

    from torchvision import transforms
    from torchvision.datasets import MNIST

    import enchanter.addons as addons
    from enchanter.tasks import ClassificationRunner


    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 3),
                addons.Mish(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3),
                addons.Mish(),
                nn.MaxPool2d(2)
            )
            self.fc = nn.Sequential(
                nn.Linear(64*5*5, 512),
                addons.Mish(),
                nn.Linear(512, 10)
            )

        def forward(self, x):
            out = self.conv(x)
            out = out.view(-1, 64*5*5)
            out = self.fc(out)
            return out


    train_ds = MNIST(
        "./data",
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    test_ds = MNIST(
        "./data",
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    train_loader = DataLoader(train_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)


データセットとモデルの準備が終わりました。次にSWALRに対応したRunnerを定義しましょう。
簡単のために ``ClassificationRunner`` を使いますが、``BaseRunner`` を継承した方法も流れは同じです。


::


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


最後にRunnerを定義して実行します。

::

    model = Model()
    optimizer = optim.Adam(model.parameters())

    runner = SWALRRunner(
        model, optimizer, nn.CrossEntropyLoss(),
        experiment=Experiment(),
        scheduler=[
            ExponentialLR(optimizer, gamma=0.9)
        ]
    )
    runner.add_loader("train", train_loader)
    runner.add_loader("test", test_loader)
    runner.train_config(epochs=10)
    runner.run()

