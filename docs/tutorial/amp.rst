Automatic Mixed Precision with Enchanter
=========================================

PyTorch v1.6から正式に Automatic Mixed Precisionに対応しました。
Enchanterでは、新たに導入された ``torch.cuda.amp`` モジュールを活用しより簡単にAMPを利用することができます。

.. warning::
    Mixed Precisionを活用するには対応したGPUが必要です。
    お使いのGPUの世代が Volta, Turing, Ampere などのTensor Coreを持つものであることを確認してください。


Training VGG16 using STL10 dataset
----------------------------------

簡単な例として小規模な画像分類データセットである STL 10 データセットを使ってAMPを体験してみましょう。

::

    from comet_ml import Experiment

    import torch.nn as nn
    from torch.utils.data import DataLoader

    from torchvision import transforms
    from torchvision.models import vgg16
    from torchvision.datasets import STL10


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_ds = STL10("./data", split="train", transform=transform, download=True)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=2)

    test_ds = STL10("./data", split="test", download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2)

    model = vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(4096, 10)       # 最終層の出力次元を変更


これで最低限の準備が完了しました。あとは、いつものように ``Runner`` を定義するだけですが、AMPを利用するには少し変更を加えます。

::

    from torch.cuda.amp import GradScaler                   # Import torch.cuda.amp.GradScaler
    from enchanter.wrappers import ClassificationRunner

    runner = ClassificationRunner(
        net, optimizer, criterion, Experiment()
    )

    runner.scaler = GradScaler()                            # Override

    runner.add_loader("train", train_loader)
    runner.add_loader("test", test_loader)
    runner.train_config(epochs=20)

    runner.run()

``enchanter.engine.BaseRunner`` は `scaler` というメンバー変数を持っています。
``None`` で初期化されていますが、これを ``torch.cuda.amp.GradScaler`` で上書きすることで初めて AMPが有効化されます。

ただし、カスタムRunnerを定義する場合は追加の作業が必要です。


AMP support for custom runners
--------------------------------

カスタムRunnerでAMPを有効にするには ``.scaler = GradScaler()`` に加え、
``.train_step()`` などで ``torch.cuda.amp.autocast`` が必要です。

具体的は以下のように定義します。


::

    class CustomRunner(enchanter.engine.BaseRunner):
        def __init__(self):
            super().__init__()
            self.model = model
            self.optimizer = optimizer
            self.criterion = nn.CrossEntropyLoss()

            self.experiment = Experiment()

            self.scaler = torch.cuda.amp.GradScaler()       # REQUIRED

        def train_step(self, batch):
            x, y = batch

            with torch.cuda.amp.autocast():                 # REQUIRED
                out = self.model(x)
                loss = self.criterion(out, y)

            return {"loss": loss}



実装のヒントは `PyTorch公式の資料 <https://pytorch.org/docs/stable/amp.html>`_ を参考にしてください。
