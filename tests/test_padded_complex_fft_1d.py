"""Tests for padded complex-to-complex 1D FFTs using cuFFTDx, comparing against PyTorch."""

import torch
import zipfft

import pytest

# Skip entire module if padded_cfft1d is not available
pytestmark = pytest.mark.skipif(
    not zipfft.is_extension_available("padded_cfft1d"),
    reason="padded_cfft1d extension not available",
)


if zipfft.padded_cfft1d is not None:
    FORWARD_FFT_CONFIGS = [
        (fft_size, signal_length, batch_size)
        for fft_size, signal_length, batch_size, is_forward in zipfft.padded_cfft1d.get_supported_configs()
        if is_forward
    ]
    INVERSE_FFT_CONFIGS = [
        (fft_size, signal_length, batch_size)
        for fft_size, signal_length, batch_size, is_forward in zipfft.padded_cfft1d.get_supported_configs()
        if not is_forward
    ]
else:
    FORWARD_FFT_CONFIGS = []
    INVERSE_FFT_CONFIGS = []

DATA_TYPES = [torch.complex64]


def run_padded_forward_cfft_test(
    fft_size: int,
    signal_length: int,
    batch_size: int,
    dtype: torch.dtype = torch.complex64,
):
    """Runs a padded forward complex FFT test for the cuFFTDx backend against PyTorch.

    Parameters
    ----------
    fft_size : int
        The size of the FFT to run, including the zero-padding.
    signal_length : int
        The length of the input signal to be padded to `fft_size`.
    batch_size : int
        The number of signals in the batch. If `batch_size > 1`, the first dimension of
        the input tensor is treated as the batch size.
    dtype : torch.dtype, optional
        The data type of the input tensor, by default torch.complex64.
    """
    # Create input of the specified signal length and batch size
    if batch_size == 1:
        x_in = torch.randn(signal_length, dtype=dtype, device="cuda")
    else:
        x_in = torch.randn(batch_size, signal_length, dtype=dtype, device="cuda")

    x_in_copy = x_in.clone()

    # PyTorch reference: pad input to fft_size, then fft
    if batch_size == 1:
        x_in_padded = torch.nn.functional.pad(x_in, (0, fft_size - signal_length))
    else:
        x_in_padded = torch.nn.functional.pad(x_in, (0, fft_size - signal_length))

    x_out_torch = torch.fft.fft(x_in_padded, n=fft_size, dim=-1)

    # Truncate back to original signal length for comparison with in-place operation
    if batch_size == 1:
        x_out_torch = x_out_torch[:signal_length]
    else:
        x_out_torch = x_out_torch[..., :signal_length]

    # Our implementation (in-place operation)
    zipfft.padded_cfft1d.pcfft(x_in_copy, fft_size)

    assert torch.allclose(x_out_torch, x_in_copy, atol=1e-4, rtol=1e-4), (
        f"Padded forward FFT results do not match ground truth for "
        f"signal_length={signal_length}, fft_size={fft_size}, batch_size={batch_size}"
    )


def run_padded_inverse_cfft_test(
    fft_size: int,
    signal_length: int,
    batch_size: int,
    dtype: torch.dtype = torch.complex64,
):
    """Runs a padded inverse complex FFT test for the cuFFTDx backend against PyTorch.

    Parameters
    ----------
    fft_size : int
        The size of the FFT to run, including the zero-padding.
    signal_length : int
        The length of the input signal to be padded to `fft_size`.
    batch_size : int
        The number of signals in the batch. If `batch_size > 1`, the first dimension of
        the input tensor is treated as the batch size.
    dtype : torch.dtype, optional
        The data type of the input tensor, by default torch.complex64.
    """
    # Create input of the specified signal length and batch size
    if batch_size == 1:
        x_in = torch.randn(signal_length, dtype=dtype, device="cuda")
    else:
        x_in = torch.randn(batch_size, signal_length, dtype=dtype, device="cuda")

    x_in_copy = x_in.clone()

    # PyTorch reference: pad input to fft_size, then ifft
    if batch_size == 1:
        x_in_padded = torch.nn.functional.pad(x_in, (0, fft_size - signal_length))
    else:
        x_in_padded = torch.nn.functional.pad(x_in, (0, fft_size - signal_length))

    x_out_torch = torch.fft.ifft(x_in_padded, dim=-1, norm="forward")

    # Truncate back to original signal length for comparison with in-place operation
    if batch_size == 1:
        x_out_torch = x_out_torch[:signal_length]
    else:
        x_out_torch = x_out_torch[..., :signal_length]

    # Our implementation (in-place operation)
    zipfft.padded_cfft1d.picfft(x_in_copy, fft_size)

    assert torch.allclose(x_out_torch, x_in_copy, atol=1e-4, rtol=1e-4), (
        f"Padded inverse FFT results do not match ground truth for "
        f"signal_length={signal_length}, fft_size={fft_size}, batch_size={batch_size}"
    )


@pytest.mark.parametrize("fft_size,signal_length,batch_size", FORWARD_FFT_CONFIGS)
@pytest.mark.parametrize("dtype", DATA_TYPES)
def test_padded_fft_c2c_1d_forward(fft_size, signal_length, batch_size, dtype):
    """Test padded forward FFT for specific signal length, fft_size, and dtype."""
    run_padded_forward_cfft_test(fft_size, signal_length, batch_size, dtype)


@pytest.mark.parametrize("fft_size,signal_length,batch_size", INVERSE_FFT_CONFIGS)
@pytest.mark.parametrize("dtype", DATA_TYPES)
def test_padded_fft_c2c_1d_inverse(fft_size, signal_length, batch_size, dtype):
    """Test padded inverse FFT for specific signal length, fft_size, and dtype."""
    run_padded_inverse_cfft_test(fft_size, signal_length, batch_size, dtype)


def test_padded_cfft_tensor_validation():
    """Test input validation for padded complex FFT functions."""
    if not zipfft.is_extension_available("padded_cfft1d"):
        pytest.skip("padded_cfft1d extension not available")

    # Test CPU tensor rejection
    x_cpu = torch.randn(32, dtype=torch.complex64)
    with pytest.raises(RuntimeError, match="Data tensor must be on CUDA device"):
        zipfft.padded_cfft1d.pcfft(x_cpu, 64)

    # Test wrong dtype rejection
    x_wrong_dtype = torch.randn(32, dtype=torch.float32, device="cuda")
    with pytest.raises(
        RuntimeError, match="Data tensor must be of type torch.complex64"
    ):
        zipfft.padded_cfft1d.pcfft(x_wrong_dtype, 64)

    # Test signal length > fft_size rejection
    x_too_long = torch.randn(128, dtype=torch.complex64, device="cuda")
    with pytest.raises(
        RuntimeError, match="Signal length must be less than or equal to FFT size"
    ):
        zipfft.padded_cfft1d.pcfft(x_too_long, 64)

    # Test unsupported configuration
    x_valid = torch.randn(7, dtype=torch.complex64, device="cuda")  # Unusual size
    with pytest.raises(RuntimeError, match="Unsupported.*FFT configuration"):
        zipfft.padded_cfft1d.pcfft(x_valid, 13)  # Unusual FFT size


def test_padded_cfft_shape_handling():
    """Test that both 1D and 2D tensors are handled correctly."""
    if not zipfft.is_extension_available("padded_cfft1d"):
        pytest.skip("padded_cfft1d extension not available")

    # Test 1D tensor (batch_size=1 case)
    x_1d = torch.randn(16, dtype=torch.complex64, device="cuda")
    x_1d_copy = x_1d.clone()

    # Should not raise an error
    zipfft.padded_cfft1d.pcfft(x_1d, 64)

    # Test 2D tensor
    x_2d = torch.randn(1, 16, dtype=torch.complex64, device="cuda")
    x_2d_copy = x_2d.clone()

    # Should not raise an error
    zipfft.padded_cfft1d.pcfft(x_2d, 64)

    # Test 3D tensor (should fail)
    x_3d = torch.randn(1, 1, 16, dtype=torch.complex64, device="cuda")
    with pytest.raises(RuntimeError, match="Data tensor must be 1D or 2D"):
        zipfft.padded_cfft1d.pcfft(x_3d, 64)
