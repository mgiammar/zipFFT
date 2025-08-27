"""Tests for padded real-to-complex 1D FFTs using cuFFTDx, comparing against PyTorch."""

import torch
from zipfft import padded_rfft1d

import pytest
import yaml
import os

FORWARD_FFT_CONFIGS = padded_rfft1d.get_supported_configs()
BATCH_SCALE_FACTOR = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
DATA_TYPES = [torch.float32]


def run_padded_forward_rfft_test(
    fft_size: int,
    signal_length: int,
    batch_size: int,
    dtype: torch.dtype = torch.float32,
):
    """Runs a padded forward FFT test for the cuFFTDx backend against PyTorch.

    Parameters
    ----------
    fft_size : int
        The size of the FFT to run, including the zero-padding.
    signal_length : int
        The length of the input signal to be padded to `fft_size`.
    batch_size : int
        The number of signals in the batch. If `batch_size > 1`, the first dimension of
        the input tensor is treated as the batch size.
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

    # Create input of the specified signal length and batch size
    x_in = torch.randn(batch_size, signal_length, dtype=dtype, device="cuda")
    x_in_copy = x_in.clone()

    # Output shape for rfft with n=fft_size
    x_out = torch.empty(
        (batch_size, fft_size // 2 + 1), dtype=complex_dtype, device="cuda"
    )
    x_out_copy = x_out.clone()

    # PyTorch reference: pad input to fft_size, then rfft
    x_in_padded = torch.nn.functional.pad(x_in, (0, fft_size - signal_length))
    torch.fft.rfft(x_in_padded, n=fft_size, out=x_out, dim=-1)

    # Our implementation
    padded_rfft1d.prfft(x_in_copy, x_out_copy, fft_size)

    assert torch.allclose(x_out, x_out_copy, atol=1e-4), (
        f"Padded FFT results do not match ground truth for "
        f"signal_length={signal_length}, fft_size={fft_size}, batch_size={batch_size}"
    )

@pytest.mark.parametrize("fft_size,batch_size", FORWARD_FFT_CONFIGS)
@pytest.mark.parametrize("dtype", DATA_TYPES)
@pytest.mark.parametrize("batch_scale", BATCH_SCALE_FACTOR)
def test_padded_fft_r2c_1d(fft_size, batch_size, dtype, batch_scale):
    """Test padded forward FFT for specific signal length, fft_size, and dtype."""
    # generate a random signal length less than or equal to fft_size
    signal_length = torch.randint(1, fft_size + 1, (1,)).item()
    run_padded_forward_rfft_test(fft_size, signal_length, batch_scale, dtype)
