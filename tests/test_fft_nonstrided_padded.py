"""Tests for padded real-to-complex 1D FFTs using cuFFTDx, comparing against PyTorch."""

import torch
from zipfft import fft_nonstrided_padded

import pytest
import yaml
import os

FORWARD_FFT_CONFIGS = fft_nonstrided_padded.get_supported_sizes()
BATCH_SCALE_FACTOR = list(range(1, 20))
DATA_TYPES = [torch.complex64]


def run_padded_forward_fft_test(
    fft_size: int,
    signal_length: int,
    batch_size: int,
    dtype: torch.dtype = torch.complex64,
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

    fft_shape = (batch_size, fft_size)

    x0 = torch.randn(fft_shape, dtype=dtype, device="cuda")

    x1 = x0.clone()

    x0[:, signal_length:] = 0

    torch.fft.fft(x0, out=x0, dim=-1)

    # Our implementation
    fft_nonstrided_padded.fft(x1, signal_length)

    assert torch.allclose(x0, x1, atol=1e-3), (
        f"Padded FFT results do not match ground truth for "
        f"signal_length={signal_length}, fft_size={fft_size}, batch_size={batch_size}"
    )

def run_padded_layered_fft_test(
    fft_size: int,
    signal_length: int,
    layer_count: int,
    batch_size1: int,
    batch_size2: int,
    dtype: torch.dtype = torch.complex64,
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

    fft_shape = (batch_size1, batch_size2, fft_size)

    x0 = torch.randn(fft_shape, dtype=dtype, device="cuda")

    x1 = x0.clone()

    x0[:, :, signal_length:] = 0

    torch.fft.fft(x0, out=x0, dim=-1)

    x0[:, layer_count:, :] = x1[:, layer_count:, :]

    # Our implementation
    fft_nonstrided_padded.fft_layered(x1, signal_length, layer_count)

    assert torch.allclose(x0, x1, atol=1e-3), (
        f"Padded FFT results do not match ground truth for "
        f"signal_length={signal_length}, fft_size={fft_size}, batch_size1={batch_size1}, batch_size2={batch_size2}, layer_count={layer_count}"
    )

@pytest.mark.parametrize("fft_size", FORWARD_FFT_CONFIGS)
@pytest.mark.parametrize("dtype", DATA_TYPES)
@pytest.mark.parametrize("batch_scale", BATCH_SCALE_FACTOR)
def test_padded_fft_c2c_1d(fft_size, dtype, batch_scale):
    """Test padded forward FFT for specific signal length, fft_size, and dtype."""
    # generate a random signal length less than or equal to fft_size
    signal_length = torch.randint(1, fft_size + 1, (1,)).item()
    run_padded_forward_fft_test(fft_size, signal_length, batch_scale, dtype)

@pytest.mark.parametrize("fft_size", FORWARD_FFT_CONFIGS)
@pytest.mark.parametrize("dtype", DATA_TYPES)
@pytest.mark.parametrize("batch_scale1", BATCH_SCALE_FACTOR)
@pytest.mark.parametrize("batch_scale2", BATCH_SCALE_FACTOR)
def test_padded_layered_fft_c2c_1d(fft_size, dtype, batch_scale1, batch_scale2):
    """Test padded forward FFT for specific signal length, fft_size, and dtype."""
    # generate a random signal length less than or equal to fft_size
    signal_length = torch.randint(1, fft_size + 1, (1,)).item()
    layer_count = torch.randint(1, batch_scale2 + 1, (1,)).item()
    run_padded_layered_fft_test(fft_size, signal_length, layer_count, batch_scale1, batch_scale2, dtype)
