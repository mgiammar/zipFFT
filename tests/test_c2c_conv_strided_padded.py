"""Simple complex-to-complex 1D FFT tests for cuFFTDx comparing against PyTorch."""

import torch  # !!! NOTE !!! CUDA backend built by PyTorch needs torch imported first!
from zipfft import conv1d_strided_padded

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

ALL_CONFIGS = conv1d_strided_padded.get_supported_configs()
BATCH_SCALE_FACTOR = [1, 2, 3, 4, 5, 6]
OUTER_BATCH_SCALE = [1, 2, 3, 4, 5, 6]
DATA_TYPES = [torch.complex64]

def run_convolution_strided_padded_test(fft_shape: int, dtype: torch.dtype, signal_length: int):
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

    x0[:, signal_length:, :] = 0

    torch.fft.fft(x0, out=x0, dim=-2)
    x0 *= kernel
    torch.fft.ifft(x0, out=x0, dim=-2)
    x0 *= float(fft_shape[-2]) 

    # NOTE: This zipFFT function is in-place
    conv1d_strided_padded.conv(x1, kernel, signal_length)

    assert torch.allclose(x0, x1, atol=5e-3), "FFT results do not match ground truth"


@pytest.mark.parametrize("fft_size,batch_size", ALL_CONFIGS)
@pytest.mark.parametrize("dtype", DATA_TYPES)
@pytest.mark.parametrize("batch_scale", BATCH_SCALE_FACTOR)
@pytest.mark.parametrize("outer_batch_scale", OUTER_BATCH_SCALE)
def test_convolution_strided(fft_size, batch_size, dtype, batch_scale, outer_batch_scale):
    """Test forward FFT for specific size, batch size, and dtype."""
    shape = (outer_batch_scale, fft_size, batch_size * batch_scale) if batch_size > 1 else (outer_batch_scale, fft_size, batch_scale)
    signal_length = torch.randint(1, fft_size + 1, (1,)).item()
    run_convolution_strided_padded_test(fft_shape=shape, dtype=dtype, signal_length=signal_length)

