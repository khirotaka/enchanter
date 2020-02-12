import torch
import torch.jit
import enchanter.addon as addon


def main():
    x = torch.tensor([10, 20, 30, 40], requires_grad=True, dtype=torch.float32)
    activation = torch.jit.trace(addon.Swish(beta=True), (x, ))

    out = activation(x)
    out_sum = out.sum()
    out_sum.backward()

    print(out)
    print(out_sum)
    print(x.grad)


if __name__ == '__main__':
    main()
