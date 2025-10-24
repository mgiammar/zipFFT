"""Simple complex-to-complex 1D FFT tests for cuFFTDx comparing against PyTorch."""

import torch  # !!! NOTE !!! CUDA backend built by PyTorch needs torch imported first!
from zipfft import fft_strided

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

ALL_CONFIGS = fft_strided.get_supported_sizes()
BATCH_SCALE_FACTOR = [1, 2, 3, 4, 5, 6]
OUTER_BATCH_SCALE = [1, 2, 3, 4, 5, 6]
DATA_TYPES = [torch.complex64]
SMEM_TRANSPOSE_OPTIONS = [True, False]

def run_forward_fft_test(fft_shape: int, dtype: torch.dtype, smem_transpose: bool):
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

    x0 = torch.fft.fft(x0, dim=1)

    # NOTE: This zipFFT function is in-place
    fft_strided.fft(x1, smem_transpose)

    # convert torch tensors to numpy arrays

    x0_numpy = x0.cpu().numpy()
    x1_numpy = x1.cpu().numpy()

    #print("Torch FFT result (first element):", x0_numpy.shape)
    #print("zipFFT FFT result (first element):", x1_numpy.shape)

    # save results to a .npy file for further analysis

    #np.save("torch_fft_result.npy", x0_numpy[0])
    #np.save("zipfft_fft_result.npy", x1_numpy[0])

    # save diffs to a .npy file for further analysis

    # np.save("fft_diff.npy", x0_numpy[0] - x1_numpy[0])

    assert torch.allclose(x0, x1, atol=1e-4), "FFT results do not match ground truth"

def run_inverse_fft_test(fft_shape: int, dtype: torch.dtype, smem_transpose: bool):
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

    torch.fft.ifft(x0, out=x0, dim=-2)
    x0 *= float(fft_shape[-2])  # Scale the output to match the inverse FFT definition

    # NOTE: This zipFFT function is in-place
    fft_strided.ifft(x1, smem_transpose)

    x0_numpy = x0.cpu().numpy()
    x1_numpy = x1.cpu().numpy()

    #print("Torch FFT result (first element):", x0_numpy.shape)
    #print("zipFFT FFT result (first element):", x1_numpy.shape)

    # save results to a .npy file for further analysis

    #np.save("torch_fft_result.npy", x0_numpy[0])
    #np.save("zipfft_fft_result.npy", x1_numpy[0])

    # save diffs to a .npy file for further analysis

    #np.save("fft_diff.npy", x0_numpy[0] - x1_numpy[0])
    #np.save("fft_scale.npy", x0_numpy[0] / x1_numpy[0])

    assert torch.allclose(x0, x1, atol=1e-4), "FFT results do not match ground truth"


@pytest.mark.parametrize("fft_size", ALL_CONFIGS)
@pytest.mark.parametrize("dtype", DATA_TYPES)
@pytest.mark.parametrize("batch_scale", BATCH_SCALE_FACTOR)
@pytest.mark.parametrize("outer_batch_scale", OUTER_BATCH_SCALE)
@pytest.mark.parametrize("smem_transpose", SMEM_TRANSPOSE_OPTIONS)
def test_fft_c2c_1d(fft_size, dtype, batch_scale, outer_batch_scale, smem_transpose):
    """Test forward FFT for specific size, batch size, and dtype."""
    shape = (outer_batch_scale, fft_size, batch_scale)
    run_forward_fft_test(fft_shape=shape, dtype=dtype, smem_transpose=smem_transpose)

@pytest.mark.parametrize("fft_size", ALL_CONFIGS)
@pytest.mark.parametrize("dtype", DATA_TYPES)
@pytest.mark.parametrize("batch_scale", BATCH_SCALE_FACTOR)
@pytest.mark.parametrize("outer_batch_scale", OUTER_BATCH_SCALE)
@pytest.mark.parametrize("smem_transpose", SMEM_TRANSPOSE_OPTIONS)
def test_ifft_c2c_1d(fft_size, dtype, batch_scale, outer_batch_scale, smem_transpose):
    """Test inverse FFT for specific size, batch size, and dtype."""
    shape = (outer_batch_scale, fft_size, batch_scale)
    run_inverse_fft_test(fft_shape=shape, dtype=dtype, smem_transpose=smem_transpose)
