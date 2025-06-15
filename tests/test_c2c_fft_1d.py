"""Simple complex-to-complex 1D FFT tests for cuFFTDx comparing against PyTorch."""

import torch  # !!! NOTE !!! CUDA backend built by PyTorch needs torch imported first!
from zipfft import zipfft_binding

import pytest

FFT_SIZES = [16, 32, 64, 128, 256, 512, 1024]
FFT_DTYPES = [torch.complex64]


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


@pytest.mark.parametrize("fft_size", FFT_SIZES)
@pytest.mark.parametrize("dtype", FFT_DTYPES)
def test_fft_c2c_1d(fft_size, dtype):
    """Test forward FFT for specific size and dtype."""
    run_forward_fft_test(fft_size, dtype)


@pytest.mark.parametrize("fft_size", FFT_SIZES)
@pytest.mark.parametrize("dtype", FFT_DTYPES)
def test_ifft_c2c_1d(fft_size, dtype):
    """Test inverse FFT for specific size and dtype."""
    run_inverse_fft_test(fft_size, dtype)
