"""Simple complex-to-complex 1D FFT tests for cuFFTDx comparing against PyTorch."""

import torch  # !!! NOTE !!! CUDA backend built by PyTorch needs torch imported first!
from zipfft import fft_nonstrided

import pytest
import yaml
import os

import numpy as np

from matplotlib import pyplot as plt

TYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex32": torch.complex32,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}

SIZES = fft_nonstrided.get_supported_sizes()
BATCH_SCALE_FACTOR = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
DATA_TYPES = [torch.complex64]

def run_forward_fft_test(fft_shape: int, dtype: torch.dtype, direction: bool):
    """Runs a single forward FFT test for a given size and dtype.

    Parameters
    ----------
    fft_shape : int
        The size of the FFT to run, first dimension is batch size if > 1.
        If a single integer is provided, it is treated as the size of the FFT.
    dtype : torch.dtype, optional
        The data type of the input tensor, by default torch.complex64.
    direction : bool
        The direction of the FFT, True for inverse FFT, False for forward FFT.
    """
    x0 = torch.randn(fft_shape, dtype=dtype, device="cuda")
    x1 = x0.clone()

    if direction:
        torch.fft.ifft(x0, out=x0)
        x0 *= float(fft_shape[-1])
    else:
        torch.fft.fft(x0, out=x0)

    # NOTE: This zipFFT function is in-place
    fft_nonstrided.fft(x1, direction)

    # convert torch tensors to numpy arrays

    x0_numpy = x0.cpu().numpy()
    x1_numpy = x1.cpu().numpy()

    print("Torch FFT result (first element):", x0_numpy.shape)
    print("zipFFT FFT result (first element):", x1_numpy.shape)

    assert torch.allclose(x0, x1, atol=1e-4), "FFT results do not match ground truth"


@pytest.mark.parametrize("fft_size", SIZES)
@pytest.mark.parametrize("dtype", DATA_TYPES)
@pytest.mark.parametrize("batch_scale", BATCH_SCALE_FACTOR)
@pytest.mark.parametrize("direction", [False, True])
def test_fft_c2c_1d(fft_size, dtype, batch_scale, direction):
    """Test forward FFT for specific size, batch size, and dtype."""
    shape = (batch_scale, fft_size)
    run_forward_fft_test(fft_shape=shape, dtype=dtype, direction=direction)

