"""Simple real-to-complex 1D FFT tests for the cuFFTDx comparing against PyTorch."""

import torch  # !!! NOTE !!! CUDA backend built by PyTorch needs torch imported first!
from zipfft import zipfft_binding

import pytest

FFT_SIZES = [16, 32, 64, 128, 256, 512, 1024]
FFT_DTYPES = [torch.float32]


def run_inverse_rfft_test(fft_size: int, dtype: torch.dtype = torch.float32):
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
    
    x_in = torch.randn((fft_size // 2 + 1,), dtype=complex_dtype, device="cuda")
    x_in_copy = x_in.clone()
    
    x_out = torch.empty(fft_size, dtype=dtype, device="cuda")
    x_out_copy = x_out.clone()
    
    torch.fft.irfft(x_in, out=x_out)
    x_out *= float(fft_size)  # Scale the output to match the inverse FFT definition
    
    zipfft_binding.ifft_c2r_1d(x_in_copy, x_out_copy)
    assert torch.allclose(x_out, x_out_copy, atol=1e-4), "FFT results do not match ground truth"
    
    

@pytest.mark.parametrize("fft_size", FFT_SIZES)
@pytest.mark.parametrize("dtype", FFT_DTYPES)
def test_fft_c2r_1d(fft_size, dtype):
    """Test forward FFT for specific size and dtype."""
    run_inverse_rfft_test(fft_size, dtype)