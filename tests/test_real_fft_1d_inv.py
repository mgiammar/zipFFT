"""Simple complex-to-real 1D FFT tests for the cuFFTDx comparing against PyTorch."""

import torch  # !!! NOTE !!! CUDA backend built by PyTorch needs torch imported first!
import zipfft

import pytest
import yaml
import os

# Skip entire module if rfft1d is not available
pytestmark = pytest.mark.skipif(
    not zipfft.is_extension_available("rfft1d"), reason="rfft1d extension not available"
)


if zipfft.rfft1d is not None:
    INVERSE_FFT_CONFIGS = zipfft.rfft1d.get_supported_configs()
else:
    INVERSE_FFT_CONFIGS = []
DATA_TYPES = [torch.float32]


def run_inverse_rfft_test(fft_shape: int, dtype: torch.dtype = torch.float32):
    """Runs a single inverse FFT test for a given size and dtype.

    Parameters
    ----------
    fft_shape : int
        The size of the FFT to run.
    dtype : torch.dtype, optional
        The data type of the output tensor, by default torch.float32.
    """
    # Mapping from the real-type back to complex type
    if dtype is torch.float16:
        complex_dtype = torch.complex32
    elif dtype is torch.float32:
        complex_dtype = torch.complex64
    elif dtype is torch.float64:
        complex_dtype = torch.complex128
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Complex shape logic
    if len(fft_shape) == 1:
        complex_fft_shape = (fft_shape[0] // 2 + 1,)
    else:
        complex_fft_shape = (fft_shape[0], fft_shape[1] // 2 + 1)

    x_in = torch.randn(complex_fft_shape, dtype=complex_dtype, device="cuda")
    x_in_copy = x_in.clone()

    x_out = torch.empty(fft_shape, dtype=dtype, device="cuda")
    x_out_copy = x_out.clone()

    torch.fft.irfft(x_in, out=x_out)
    x_out *= float(
        fft_shape[-1]
    )  # Scale the output to match the inverse FFT definition

    zipfft.rfft1d.irfft(x_in_copy, x_out_copy)
    assert torch.allclose(
        x_out, x_out_copy, atol=1e-4
    ), "FFT results do not match ground truth"


@pytest.mark.parametrize("fft_size,batch_size", INVERSE_FFT_CONFIGS)
@pytest.mark.parametrize("dtype", DATA_TYPES)
def test_fft_c2r_1d(fft_size, batch_size, dtype):
    """Test inverse FFT for specific size, batch size, and dtype."""
    shape = (batch_size, fft_size) if batch_size > 1 else (fft_size,)
    run_inverse_rfft_test(fft_shape=shape, dtype=dtype)
