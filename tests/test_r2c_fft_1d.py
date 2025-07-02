"""Simple real-to-complex 1D FFT tests for the cuFFTDx comparing against PyTorch."""

import torch  # !!! NOTE !!! CUDA backend built by PyTorch needs torch imported first!
from zipfft import zipfft_binding

import pytest
import yaml
import os

# Load FFT config from YAML file
FFT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../configs/fft_r2c_1d.yaml")
TYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex32": torch.complex32,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}

with open(FFT_CONFIG_PATH, "r") as f:
    config_list = yaml.safe_load(f)

# Parse forward FFT (R2C) configurations
forward_configs = [cfg for cfg in config_list if cfg["is_forward_fft"]]

# Extract unique sizes and types for forward FFTs
FORWARD_FFT_SIZES = sorted(set(cfg["fft_size"] for cfg in forward_configs))
FORWARD_FFT_TYPES = sorted(
    set(TYPE_MAP[cfg["input_data_type"]] for cfg in forward_configs),
    key=lambda x: str(x)
)


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

    zipfft_binding.fft_r2c_1d(x_in_copy, x_out_copy)
    assert torch.allclose(
        x_out, x_out_copy, atol=1e-4
    ), "FFT results do not match ground truth"


@pytest.mark.parametrize("fft_size", FORWARD_FFT_SIZES)
@pytest.mark.parametrize("dtype", FORWARD_FFT_TYPES)
def test_fft_r2c_1d(fft_size, dtype):
    """Test forward FFT for specific size and dtype."""
    run_forward_rfft_test(fft_size, dtype)
