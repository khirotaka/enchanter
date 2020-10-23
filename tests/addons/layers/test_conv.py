import torch
import torch.jit
import numpy as np
import enchanter.addons.layers as layers


def test_causalconv_1():
    x = torch.randn(1, 6, 128, requires_grad=True)     # [N, C, L]
    model = layers.CausalConv1d(6, 32, 3)

    out = model(x)
    out = out.sum()
    out.backward()
    assert x.grad is not None


def compute_causalconv1d(x: np.ndarray, kernels: np.ndarray, dilation: int) -> np.ndarray:
    """
    References:
        https://github.com/awslabs/gluon-ts/blob/master/test/model/seq2seq/test_cnn.py#L23

    """

    conv_x = np.zeros_like(x)
    # compute in a naive way.
    for (t, xt) in enumerate(x):
        dial_offset = 0
        for i in reversed(range(len(kernels))):
            xt_lag = x[t - dial_offset] if t - dial_offset >= 0 else 0.0
            dial_offset += dilation
            conv_x[t] += kernels[i] * xt_lag

    return conv_x


def test_causalconv_2():
    x = np.random.normal(0, 1, size=(1, 1, 10)).astype(np.float32)
    x = torch.from_numpy(x)

    conv1d = torch.jit.script(layers.CausalConv1d(1, 1, kernel_size=3, dilation=3))
    torch.nn.init.ones_(conv1d.weight)
    torch.nn.init.zeros_(conv1d.bias)

    y1 = conv1d(x).reshape(-1).detach().numpy()

    y2 = compute_causalconv1d(
        x.reshape(-1).numpy(),
        kernels=np.array([1.0] * 3),
        dilation=3
    )

    assert (np.max(np.abs(y1 - y2)) < 1e-5)
