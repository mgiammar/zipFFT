"""Simple complex-to-complex 1D FFT tests for cuFFTDx comparing against PyTorch."""

import torch  # !!! NOTE !!! CUDA backend built by PyTorch needs torch imported first!
from zipfft import conv1d_strided

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

ALL_CONFIGS = conv1d_strided.get_supported_configs()
BATCH_SCALE_FACTOR = [1, 2, 3, 4, 5, 6]
OUTER_BATCH_SCALE = [1, 2, 3, 4, 5, 6]
DATA_TYPES = [torch.complex64]

def run_forward_fft_test(fft_shape: int, dtype: torch.dtype = torch.complex64):
    """Runs a single forward FFT test for a given size and dtype.

    Parameters
    ----------
    fft_shape : int
        The size of the FFT to run, first dimension is batch size if > 1.
        If a single integer is provided, it is treated as the size of the FFT.
    dtype : torch.dtype, optional
        The data type of the input tensor, by default torch.complex64.
    """
    # make a square signal in x0
    x0 = torch.zeros(fft_shape, dtype=dtype, device="cuda")
    x0[..., fft_shape[-1]//4:3*fft_shape[-1]//4,
         fft_shape[-2]//4:3*fft_shape[-2]//4] = 1.0
    
    # make a circular kernel

    kernel = torch.zeros(fft_shape, dtype=dtype, device="cuda")
    y, x = torch.meshgrid(torch.arange(fft_shape[-2], device="cuda") - fft_shape[-2]//2,
                          torch.arange(fft_shape[-1], device="cuda") - fft_shape[-1]//2,
                          indexing='ij')
    radius = fft_shape[-2]//8
    mask = x**2 + y**2 <= radius**2
    kernel[..., mask] = 1.0
    kernel /= kernel.sum()  # normalize kernel

    x1 = x0.clone()

    torch.fft.fft(x0, out=x0, dim=-2)
    x0 *= kernel
    torch.fft.ifft(x0, out=x0, dim=-2)
    x0 *= float(fft_shape[-2]) 

    # NOTE: This zipFFT function is in-place
    conv1d_strided.conv(x1, kernel)

    assert torch.allclose(x0, x1, atol=1e-3), "FFT results do not match ground truth"


#run_forward_fft_test(fft_shape=(16, 1024, 1024), dtype=torch.complex64)

@pytest.mark.parametrize("fft_size,batch_size", ALL_CONFIGS)
@pytest.mark.parametrize("dtype", DATA_TYPES)
@pytest.mark.parametrize("batch_scale", BATCH_SCALE_FACTOR)
@pytest.mark.parametrize("outer_batch_scale", OUTER_BATCH_SCALE)
def test_fft_c2c_1d(fft_size, batch_size, dtype, batch_scale, outer_batch_scale):
    """Test forward FFT for specific size, batch size, and dtype."""
    shape = (outer_batch_scale, fft_size, batch_size * batch_scale) if batch_size > 1 else (outer_batch_scale, fft_size, batch_scale)
    run_forward_fft_test(fft_shape=shape, dtype=dtype)

