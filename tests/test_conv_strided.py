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
DO_KERNEL_TRANSPOSE = [True, False]
DO_SMEM_TRANSPOSE = [True, False]

def run_convolution_strided_test(fft_shape: int, dtype: torch.dtype, smem_transpose, kernel_transpose):
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
    zipfft_kernel = kernel

    if kernel_transpose:
        kernel_shape = (conv_strided.kernel_size(x0),)
        zipfft_kernel = torch.randn(kernel_shape, dtype=dtype, device="cuda")
        conv_strided.kernel_transpose(kernel, zipfft_kernel)

    x1 = x0.clone()

    torch.fft.fft(x0, out=x0, dim=-2)
    x0 *= kernel
    torch.fft.ifft(x0, out=x0, dim=-2)
    x0 *= float(fft_shape[-2]) 

    # NOTE: This zipFFT function is in-place
    conv_strided.conv(x1, zipfft_kernel, kernel_transpose, smem_transpose)

    assert torch.allclose(x0, x1, atol=5e-3), "FFT results do not match ground truth"


@pytest.mark.parametrize("fft_size", ALL_CONFIGS)
@pytest.mark.parametrize("dtype", DATA_TYPES)
@pytest.mark.parametrize("batch_scale", BATCH_SCALE_FACTOR)
@pytest.mark.parametrize("outer_batch_scale", OUTER_BATCH_SCALE)
@pytest.mark.parametrize("smem_transpose", DO_SMEM_TRANSPOSE)
@pytest.mark.parametrize("kernel_transpose", DO_KERNEL_TRANSPOSE)
def test_convolution_strided(fft_size, dtype, batch_scale, outer_batch_scale, smem_transpose, kernel_transpose):
    """Test forward FFT for specific size, batch size, and dtype."""
    shape = (outer_batch_scale, fft_size, batch_scale)
    run_convolution_strided_test(
        fft_shape=shape,
        dtype=dtype,
        smem_transpose=smem_transpose,
        kernel_transpose=kernel_transpose
    )

