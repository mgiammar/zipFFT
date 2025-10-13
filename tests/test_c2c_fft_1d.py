"""Simple complex-to-complex 1D FFT tests for cuFFTDx comparing against PyTorch."""

import torch  # !!! NOTE !!! CUDA backend built by PyTorch needs torch imported first!
import zipfft

import pytest
import yaml
import os

# Skip entire module if cfft1d is not available
pytestmark = pytest.mark.skipif(
    zipfft.is_extension_available("cfft1d"), reason="cfft1d extension not available"
)

# Only get configs if cfft1d is available
if zipfft.cfft1d is not None:
    ALL_CONFIGS = zipfft.cfft1d.get_supported_configs()
    FORWARD_FFT_CONFIGS = [(cfg[0], cfg[1]) for cfg in ALL_CONFIGS if cfg[2] is True]
    INVERSE_FFT_CONFIGS = [(cfg[0], cfg[1]) for cfg in ALL_CONFIGS if cfg[2] is False]
else:
    FORWARD_FFT_CONFIGS = []
    INVERSE_FFT_CONFIGS = []

TYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex32": torch.complex32,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}

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
    x0 = torch.randn(fft_shape, dtype=dtype, device="cuda")
    x1 = x0.clone()

    torch.fft.fft(x0, out=x0)

    # NOTE: This zipFFT function is in-place
    zipfft.cfft1d.fft(x1)

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
    zipfft.cfft1d.ifft(x1)

    assert torch.allclose(x0, x1, atol=1e-4), "FFT results do not match ground truth"


@pytest.mark.parametrize("fft_size,batch_size", FORWARD_FFT_CONFIGS)
@pytest.mark.parametrize("dtype", DATA_TYPES)
def test_fft_c2c_1d(fft_size, batch_size, dtype):
    """Test forward FFT for specific size, batch size, and dtype."""
    shape = (batch_size, fft_size) if batch_size > 1 else (fft_size,)
    run_forward_fft_test(fft_shape=shape, dtype=dtype)


@pytest.mark.parametrize("fft_size,batch_size", INVERSE_FFT_CONFIGS)
@pytest.mark.parametrize("dtype", DATA_TYPES)
def test_ifft_c2c_1d(fft_size, batch_size, dtype):
    """Test inverse FFT for specific size, batch size, and dtype."""
    shape = (batch_size, fft_size) if batch_size > 1 else (fft_size,)
    run_inverse_fft_test(fft_shape=shape, dtype=dtype)
    if not zipfft.is_extension_available("cfft1d"):
        pytest.skip("cfft1d extension not available")

    shape = (batch_size, fft_size) if batch_size > 1 else (fft_size,)
    run_inverse_fft_test(fft_shape=shape, dtype=dtype)
