"""Simple complex-to-complex 1D FFT tests for cuFFTDx comparing against PyTorch."""

import torch  # !!! NOTE !!! CUDA backend built by PyTorch needs torch imported first!
from zipfft import cfft1d

import pytest
import yaml
import os

import numpy as np

from matplotlib import pyplot as plt

TYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex32": torch.complex32,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}

ALL_CONFIGS = cfft1d.get_supported_configs()
FORWARD_FFT_CONFIGS = [(cfg[0], cfg[1]) for cfg in ALL_CONFIGS if cfg[2] is True]
INVERSE_FFT_CONFIGS = [(cfg[0], cfg[1]) for cfg in ALL_CONFIGS if cfg[2] is False]
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
    cfft1d.fft(x1)

    # convert torch tensors to numpy arrays

    x0_numpy = x0.cpu().numpy()
    x1_numpy = x1.cpu().numpy()

    print("Torch FFT result (first element):", x0_numpy.shape)
    print("zipFFT FFT result (first element):", x1_numpy.shape)

    # save the results as images for visual comparison

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Torch FFT Magnitude")
    plt.imshow(np.abs(x0_numpy), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title("zipFFT FFT Magnitude")
    plt.imshow(np.abs(x1_numpy), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.savefig("fft_comparison.png")

    np.save("torch_fft_result.npy", x0_numpy)
    np.save("zipfft_fft_result.npy", x1_numpy)

    # also show the difference

    plt.figure(figsize=(6, 6))
    plt.title("Difference in Magnitude")
    plt.imshow(np.abs(x0_numpy - x1_numpy), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.savefig("fft_difference.png")


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
    cfft1d.ifft(x1)

    assert torch.allclose(x0, x1, atol=1e-4), "FFT results do not match ground truth"


run_forward_fft_test(fft_shape=(40, 64), dtype=torch.complex64)

# @pytest.mark.parametrize("fft_size,batch_size", FORWARD_FFT_CONFIGS)
# @pytest.mark.parametrize("dtype", DATA_TYPES)
# def test_fft_c2c_1d(fft_size, batch_size, dtype):
#     """Test forward FFT for specific size, batch size, and dtype."""
#     shape = (batch_size, fft_size) if batch_size > 1 else (fft_size,)
#     run_forward_fft_test(fft_shape=shape, dtype=dtype)


# @pytest.mark.parametrize("fft_size,batch_size", INVERSE_FFT_CONFIGS)
# @pytest.mark.parametrize("dtype", DATA_TYPES)
# def test_ifft_c2c_1d(fft_size, batch_size, dtype):
#     """Test inverse FFT for specific size, batch size, and dtype."""
#     shape = (batch_size, fft_size) if batch_size > 1 else (fft_size,)
#     run_inverse_fft_test(fft_shape=shape, dtype=dtype)
