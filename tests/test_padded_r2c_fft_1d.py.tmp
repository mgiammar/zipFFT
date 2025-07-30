"""Tests for padded real-to-complex 1D FFTs using cuFFTDx, comparing against PyTorch."""

import torch
from zipfft import zipfft_binding

import pytest
import yaml
import os

# Load padded FFT config from YAML file
PADDED_FFT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "../configs/padded_fft_r2c_1d.yaml"
)
TYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex32": torch.complex32,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}

with open(PADDED_FFT_CONFIG_PATH, "r") as f:
    config_list = yaml.safe_load(f)

# Parse forward FFT (R2C) configurations
forward_configs = [cfg for cfg in config_list if cfg["is_forward_fft"]]

# Extract unique (signal_length, fft_size) pairs and types for forward FFTs
FORWARD_SIGNAL_LENGTHS = sorted(set(cfg["signal_length"] for cfg in forward_configs))
FORWARD_FFT_SIZES = sorted(set(cfg["fft_size"] for cfg in forward_configs))
FORWARD_FFT_PAIRS = sorted(
    set((cfg["signal_length"], cfg["fft_size"]) for cfg in forward_configs)
)
FORWARD_FFT_TYPES = sorted(
    set(TYPE_MAP[cfg["input_data_type"]] for cfg in forward_configs),
    key=lambda x: str(x),
)


def run_padded_forward_rfft_test(
    signal_length: int, fft_size: int, dtype: torch.dtype = torch.float32
):
    """Runs a single padded forward FFT test for a given signal length, fft_size, and dtype.

    Parameters
    ----------
    signal_length : int
        The length of the input signal (unpadded).
    fft_size : int
        The size of the FFT to run (including padding).
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

    # Create input of the specified signal length
    x_in = torch.randn(signal_length, dtype=dtype, device="cuda")
    x_in_copy = x_in.clone()

    # Output shape for rfft with n=fft_size
    x_out = torch.empty((fft_size // 2 + 1,), dtype=complex_dtype, device="cuda")
    x_out_copy = x_out.clone()

    # PyTorch reference: pad input to fft_size, then rfft
    x_in_padded = torch.nn.functional.pad(x_in, (0, fft_size - signal_length))
    torch.fft.rfft(x_in_padded, n=fft_size, out=x_out)

    # Our implementation
    zipfft_binding.padded_fft_r2c_1d(x_in_copy, x_out_copy, fft_size)

    assert torch.allclose(x_out, x_out_copy, atol=1e-4), (
        f"Padded FFT results do not match ground truth for "
        f"signal_length={signal_length}, fft_size={fft_size}"
    )


@pytest.mark.parametrize("signal_length,fft_size", FORWARD_FFT_PAIRS)
@pytest.mark.parametrize("dtype", FORWARD_FFT_TYPES)
def test_padded_fft_r2c_1d(signal_length, fft_size, dtype):
    """Test padded forward FFT for specific signal length, fft_size, and dtype."""
    run_padded_forward_rfft_test(signal_length, fft_size, dtype)
