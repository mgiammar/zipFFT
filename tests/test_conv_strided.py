"""Simple complex-to-complex 1D FFT tests for cuFFTDx comparing against PyTorch."""

import torch  # !!! NOTE !!! CUDA backend built by PyTorch needs torch imported first!
from zipfft import conv_strided

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

ALL_CONFIGS = conv_strided.get_supported_sizes()
BATCH_SCALE_FACTOR = [1, 2, 3, 4, 5, 6]
OUTER_BATCH_SCALE = [1, 2, 3, 4, 5, 6]
DATA_TYPES = [torch.complex64]

def run_convolution_strided_test(fft_shape: int, dtype: torch.dtype = torch.complex64):
    """Runs a single forward FFT test for a given size and dtype.

    Parameters
    ----------
    fft_shape : int
        The size of the FFT to run, first dimension is batch size if > 1.
        If a single integer is provided, it is treated as the size of the FFT.
    dtype : torch.dtype, optional
        The data type of the input tensor, by default torch.complex64.
    """

    # make random x0 and kernel
    x0 = torch.randn(fft_shape, dtype=dtype, device="cuda")
    kernel = torch.randn(fft_shape, dtype=dtype, device="cuda")

    x1 = x0.clone()

    torch.fft.fft(x0, out=x0, dim=-2)
    x0 *= kernel
    torch.fft.ifft(x0, out=x0, dim=-2)
    x0 *= float(fft_shape[-2]) 

    # NOTE: This zipFFT function is in-place
    conv_strided.conv(x1, kernel)

    assert torch.allclose(x0, x1, atol=5e-3), "FFT results do not match ground truth"


@pytest.mark.parametrize("fft_size", ALL_CONFIGS)
@pytest.mark.parametrize("dtype", DATA_TYPES)
@pytest.mark.parametrize("batch_scale", BATCH_SCALE_FACTOR)
@pytest.mark.parametrize("outer_batch_scale", OUTER_BATCH_SCALE)
def test_convolution_strided(fft_size, dtype, batch_scale, outer_batch_scale):
    """Test forward FFT for specific size, batch size, and dtype."""
    shape = (outer_batch_scale, fft_size, batch_scale)
    run_convolution_strided_test(fft_shape=shape, dtype=dtype)

