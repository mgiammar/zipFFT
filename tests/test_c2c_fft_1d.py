"""Simple complex-to-complex 1D FFT tests for cuFFTDx comparing against PyTorch."""

import torch  # !!! NOTE !!! CUDA backend built by PyTorch needs torch imported first!
from zipfft import zipfft_binding

import pytest
import json
import os


# Load FFT config from JSON file
FFT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../fft_sizes_config.json")
TYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex32": torch.complex32,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}

with open(FFT_CONFIG_PATH, "r") as f:
    config = json.load(f)

FORWARD_FFT_SIZES = config["forward_fft_c2c_1d"]["fft_sizes"]
FORWARD_FFT_TYPES = [TYPE_MAP[x] for x in config["forward_fft_c2c_1d"]["fft_types"]]

INVERSE_FFT_SIZES = config["inverse_fft_c2c_1d"]["fft_sizes"]
INVERSE_FFT_TYPES = [TYPE_MAP[x] for x in config["inverse_fft_c2c_1d"]["fft_types"]]



def run_forward_fft_test(fft_size: int, dtype: torch.dtype = torch.complex64):
    """Runs a single forward FFT test for a given size and dtype.

    Parameters
    ----------
    fft_size : int
        The size of the FFT to run.
    dtype : torch.dtype, optional
        The data type of the input tensor, by default torch.complex64.
    """
    x0 = torch.randn(fft_size, dtype=dtype, device="cuda")
    x1 = x0.clone()

    torch.fft.fft(x0, out=x0)

    # NOTE: This zipFFT function is in-place
    zipfft_binding.fft_c2c_1d(x1)

    assert torch.allclose(x0, x1, atol=1e-4), "FFT results do not match ground truth"


def run_inverse_fft_test(fft_size: int, dtype: torch.dtype = torch.complex64):
    """Runs a single forward FFT test for a given size and dtype.

    Parameters
    ----------
    fft_size : int
        The size of the FFT to run.
    dtype : torch.dtype, optional
        The data type of the input tensor, by default torch.complex64.
    """
    x0 = torch.randn(fft_size, dtype=dtype, device="cuda")
    x1 = x0.clone()

    torch.fft.ifft(x0, out=x0)
    x0 *= float(fft_size)  # Scale the output to match the inverse FFT definition

    # NOTE: This zipFFT function is in-place
    zipfft_binding.ifft_c2c_1d(x1)

    assert torch.allclose(x0, x1, atol=1e-4), "FFT results do not match ground truth"


@pytest.mark.parametrize("fft_size", FORWARD_FFT_SIZES)
@pytest.mark.parametrize("dtype", FORWARD_FFT_TYPES)
def test_fft_c2c_1d(fft_size, dtype):
    """Test forward FFT for specific size and dtype."""
    run_forward_fft_test(fft_size, dtype)


@pytest.mark.parametrize("fft_size", INVERSE_FFT_SIZES)
@pytest.mark.parametrize("dtype", INVERSE_FFT_TYPES)
def test_ifft_c2c_1d(fft_size, dtype):
    """Test inverse FFT for specific size and dtype."""
    run_inverse_fft_test(fft_size, dtype)
