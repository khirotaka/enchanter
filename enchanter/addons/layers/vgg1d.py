from typing import List
import torch
import torch.nn as nn


__all__ = ["VGG1d", "vgg11", "vgg13", "vgg16", "vgg19"]


class VGG1d(nn.Module):
    def __init__(
        self, extractor: nn.Module, extractor_features: int, mid_features: int, n_classes: int, init_weight: bool = True
    ) -> None:
        super(VGG1d, self).__init__()
        self.extractor = extractor
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(extractor_features, mid_features),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(mid_features, mid_features),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(mid_features, n_classes),
        )

        if init_weight:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extractor(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_extractor(in_features: int, cfg: List, batch_norm: bool = False):
    layers: List[nn.Module] = []

    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
        else:
            conv = nn.Conv1d(in_features, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv, nn.BatchNorm1d(v), nn.ReLU(True)]
            else:
                layers += [conv, nn.ReLU(True)]

            in_features = v
    return nn.Sequential(*layers)


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "C": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(in_features: int, cfg: str, batch_norm: bool, **kwargs) -> nn.Module:
    model = VGG1d(make_extractor(in_features, cfgs[cfg], batch_norm), cfgs[cfg][-2], **kwargs)
    return model


def vgg11(
    in_features: int, mid_features: int, n_classes: int, init_weight: bool = True, batch_norm: bool = True
) -> nn.Module:
    """
    Generate 1D-VGG 11 for time series.

    Args:
        in_features: the number of input features.
        mid_features: the number of nodes in fully connected layer.
        n_classes: the number of classes.
        init_weight: apply kaiming normal. default: True.
        batch_norm: apply BatchNorm1d. default: True.

    Returns:
        1D-VGG11

    """
    return _vgg(in_features, "A", batch_norm, mid_features=mid_features, n_classes=n_classes, init_weight=init_weight)


def vgg13(
    in_features: int, mid_features: int, n_classes: int, init_weight: bool = True, batch_norm: bool = True
) -> nn.Module:
    """
    Generate 1D-VGG 13 for time series.

    Args:
        in_features: the number of input features.
        mid_features: the number of nodes in fully connected layer.
        n_classes: the number of classes.
        init_weight: apply kaiming normal. default: True.
        batch_norm: apply BatchNorm1d. default: True.

    Returns:
        1D-VGG13

    """
    return _vgg(in_features, "B", batch_norm, mid_features=mid_features, n_classes=n_classes, init_weight=init_weight)


def vgg16(
    in_features: int, mid_features: int, n_classes: int, init_weight: bool = True, batch_norm: bool = True
) -> nn.Module:
    """
    Generate 1D-VGG 16 for time series.

    Args:
        in_features: the number of input features.
        mid_features: the number of nodes in fully connected layer.
        n_classes: the number of classes.
        init_weight: apply kaiming normal. default: True.
        batch_norm: apply BatchNorm1d. default: True.

    Returns:
        1D-VGG16

    """
    return _vgg(in_features, "C", batch_norm, mid_features=mid_features, n_classes=n_classes, init_weight=init_weight)


def vgg19(
    in_features: int, mid_features: int, n_classes: int, init_weight: bool = True, batch_norm: bool = True
) -> nn.Module:
    """
    Generate 1D-VGG 19 for time series.

    Args:
        in_features: the number of input features.
        mid_features: the number of nodes in fully connected layer.
        n_classes: the number of classes.
        init_weight: apply kaiming normal. default: True.
        batch_norm: apply BatchNorm1d. default: True.

    Returns:
        1D-VGG19

    """
    return _vgg(in_features, "D", batch_norm, mid_features=mid_features, n_classes=n_classes, init_weight=init_weight)
