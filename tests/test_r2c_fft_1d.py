"""Simple real-to-complex 1D FFT tests for the cuFFTDx comparing against PyTorch."""

import torch  # !!! NOTE !!! CUDA backend built by PyTorch needs torch imported first!
from zipfft import rfft1d

import pytest
import yaml
import os


FORWARD_FFT_CONFIGS = rfft1d.get_supported_configs()
DATA_TYPES = [torch.float32]


def run_forward_rfft_test(fft_size: int, dtype: torch.dtype = torch.float32):
    """Runs a single forward FFT test for a given size and dtype.

    Parameters
    ----------
    fft_size : int
        The size of the FFT to run.
    dtype : torch.dtype, optional
        The data type of the input tensor, by default torch.float32.
    """
    if dtype is torch.float16:
        complex_dtype = torch.complex32
    elif dtype is torch.float32:
        complex_dtype = torch.complex64
    elif dtype is torch.float64:
        complex_dtype = torch.complex128
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    x_in = torch.randn(fft_size, dtype=dtype, device="cuda")
    x_in_copy = x_in.clone()

    x_out = torch.empty((fft_size // 2 + 1,), dtype=complex_dtype, device="cuda")
    x_out_copy = x_out.clone()

    torch.fft.rfft(x_in, out=x_out)

    rfft1d.rfft(x_in_copy, x_out_copy)
    assert torch.allclose(
        x_out, x_out_copy, atol=1e-4
    ), "FFT results do not match ground truth"


@pytest.mark.parametrize("fft_size,batch_size", FORWARD_FFT_CONFIGS)
@pytest.mark.parametrize("dtype", DATA_TYPES)
def test_fft_r2c_1d(fft_size, batch_size, dtype):
    """Test forward FFT for specific size and dtype."""
    run_forward_rfft_test(fft_size, dtype)
