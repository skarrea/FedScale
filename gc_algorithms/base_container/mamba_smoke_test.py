"""Minimal CUDA smoke test for the compiled mamba-ssm kernels."""

import sys

import torch
from mamba_ssm import Mamba


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run this test on a Linux host with an NVIDIA "
            "GPU, a compatible driver, and the NVIDIA Container Toolkit."
        )

    device = torch.device("cuda")
    torch.manual_seed(0)

    # (batch, sequence length, feature dimension)
    x = torch.randn(2, 64, 32, device=device, requires_grad=True)
    layer = Mamba(d_model=32, d_state=16, d_conv=4, expand=2).to(device)

    y = layer(x)
    if y.shape != x.shape:
        raise AssertionError(f"Unexpected output shape: {y.shape} != {x.shape}")
    if not torch.isfinite(y).all():
        raise AssertionError("Mamba forward pass produced NaN or infinity")

    y.square().mean().backward()
    if x.grad is None or not torch.isfinite(x.grad).all():
        raise AssertionError("Mamba backward pass produced an invalid input gradient")

    parameter_gradients = [
        parameter.grad for parameter in layer.parameters() if parameter.grad is not None
    ]
    if not parameter_gradients:
        raise AssertionError("Mamba backward pass produced no parameter gradients")
    if not all(torch.isfinite(gradient).all() for gradient in parameter_gradients):
        raise AssertionError("Mamba backward pass produced an invalid parameter gradient")

    torch.cuda.synchronize()
    print(f"PASS: mamba-ssm forward and backward passes ran on {torch.cuda.get_device_name()}")
    print(f"Python {sys.version.split()[0]} | PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
    print(f"input={tuple(x.shape)} output={tuple(y.shape)} dtype={y.dtype}")


if __name__ == "__main__":
    main()
