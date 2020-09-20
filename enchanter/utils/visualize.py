import io
import sys
from typing import Tuple

import torch
import netron


__all__ = ["with_netron"]


def with_netron(
    model: torch.nn.Module,
    dummy: Tuple[torch.Tensor, ...],
    backend: str = "onnx",
    open_browser: bool = True,
    port: int = 8080,
    host: str = "localhost",
):
    """
    Visualize the PyTorch graph in a browser.

    Examples:
        >>> from enchanter.addons.layers import AutoEncoder
        >>> x = torch.randn(1, 32)  # [N, in_features]
        >>> model = AutoEncoder([32, 16, 8, 2])
        >>> with_netron(model, (x, ))

    Warnings:
        Models that cannot be graphed using TorchScript or ONNX cannot be visualized.

    Args:
        model: PyTorch model
        dummy: dummy input for generate graph
        backend: specified graph format. [torchscript or onnx]. default onnx.
        open_browser: if True, open browser.
        port: port number. default: 8080.
        host: hostname. default: localhost.

    """
    if backend.lower() == "torchscript":
        extension = "pth"
    elif backend.lower() == "onnx":
        extension = "onnx"
    else:
        raise ValueError("Unknown backend: {}".format(backend))

    model_name: str = model.__class__.__name__
    buffer = io.BytesIO()

    if extension == "pth":
        model = torch.jit.trace(model, dummy)
        torch.jit.save(model, buffer)

    else:
        torch.onnx.export(model, dummy, buffer)

    netron.serve("{}.{}".format(model_name, extension), buffer.getvalue(), browse=open_browser, port=port, host=host)
    sys.stdout.write("Press CTRL+C to quit.\n")
    netron.wait()
