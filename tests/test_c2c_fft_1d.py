"""Simple complex-to-complex 1D FFT tests for cuFFTDx comparing against PyTorch."""

import torch  # !!! NOTE !!! CUDA backend built by PyTorch needs torch imported first!
from zipfft import zipfft_binding

import pytest
import yaml
import os


TYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex32": torch.complex32,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}

# NOTE: These configurations are hard-coded. Including more instantiations in the
# complex_fft_1d_binding.cu file will require manually adding them here.
FORWARD_FFT_CONFIGS = [
    (size, batch, torch.complex64)
    for size in [64, 128, 256, 512, 1024, 2048, 4096]
    for batch in [1, 2]
]
INVERSE_FFT_CONFIGS = [
    (size, batch, torch.complex64)
    for size in [64, 128, 256, 512, 1024, 2048, 4096]
    for batch in [1, 2]
]


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
    x0 = torch.randn(fft_shape, dtype=dtype, device="cuda")
    x1 = x0.clone()

    torch.fft.fft(x0, out=x0)

    # NOTE: This zipFFT function is in-place
    zipfft_binding.fft_c2c_1d(x1)

    assert torch.allclose(x0, x1, atol=1e-4), "FFT results do not match ground truth"


def run_inverse_fft_test(fft_shape: int, dtype: torch.dtype = torch.complex64):
    """Runs a single inverse FFT test for a given size and dtype.

    Parameters
    ----------
    fft_shape : int
        The size of the FFT to run, first dimension is batch size if > 1.
        If a single integer is provided, it is treated as the size of the FFT.
    dtype : torch.dtype, optional
        The data type of the input tensor, by default torch.complex64.
    """
    x0 = torch.randn(fft_shape, dtype=dtype, device="cuda")
    x1 = x0.clone()

    torch.fft.ifft(x0, out=x0)
    x0 *= float(fft_shape[-1])  # Scale the output to match the inverse FFT definition

    # NOTE: This zipFFT function is in-place
    zipfft_binding.ifft_c2c_1d(x1)

    assert torch.allclose(x0, x1, atol=1e-4), "FFT results do not match ground truth"


@pytest.mark.parametrize("fft_size,batch_size,dtype", FORWARD_FFT_CONFIGS)
def test_fft_c2c_1d(fft_size, batch_size, dtype):
    """Test forward FFT for specific size, batch size, and dtype."""
    shape = (batch_size, fft_size) if batch_size > 1 else (fft_size,)
    run_forward_fft_test(fft_shape=shape, dtype=dtype)


@pytest.mark.parametrize("fft_size,batch_size,dtype", INVERSE_FFT_CONFIGS)
def test_ifft_c2c_1d(fft_size, batch_size, dtype):
    """Test inverse FFT for specific size, batch size, and dtype."""
    shape = (batch_size, fft_size) if batch_size > 1 else (fft_size,)
    run_inverse_fft_test(fft_shape=shape, dtype=dtype)
