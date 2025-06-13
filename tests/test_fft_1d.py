"""Simple 1D FFT executing tests for the cuFFTDx backend comparing against PyTorch."""

import torch  # !!! NOTE !!! CUDA backend built by PyTorch needs torch imported first!
from zipfft import binding_cuda

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
    binding_cuda.fft_c2c_1d(x1)

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
    binding_cuda.ifft_c2c_1d(x1)

    assert torch.allclose(x0, x1, atol=1e-4), "FFT results do not match ground truth"


def test_fft_c2c_1d():
    """Runs a series of forward FFT tests for various sizes and data types."""
    for fft_size in FFT_SIZES:
        for dtype in FFT_DTYPES:
            run_forward_fft_test(fft_size, dtype)


def test_ifft_c2c_1d():
    """Runs a series of inverse FFT tests for various sizes and data types."""
    for fft_size in FFT_SIZES:
        for dtype in FFT_DTYPES:
            run_inverse_fft_test(fft_size, dtype)
