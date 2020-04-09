import torch
import torch.optim as optim
from enchanter.addons import TransformerOptimizer


def test_transformer_optim():
    def f(x):
        return 3 * x ** 2

    a = torch.tensor(10.0, requires_grad=True).float()
    adam = TransformerOptimizer(optim.Adam([a]), warm_up=10, d_model=100)

    is_pass = False
    try:
        for i in range(100):
            adam.zero_grad()
            out = f(a)
            out.backward()
            adam.step()
    except Exception as e:
        print(e)
        is_pass = False

    else:
        is_pass = True

    assert is_pass
