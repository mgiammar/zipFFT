"""Strided padded complex-to-complex 1D FFT tests for cuFFTDx comparing against PyTorch."""

import torch  # !!! NOTE !!! CUDA backend built by PyTorch needs torch imported first!
import zipfft

import pytest

# Skip entire module if strided_padded_cfft1d is not available
pytestmark = pytest.mark.skipif(
    not zipfft.is_extension_available("strided_padded_cfft1d"),
    reason="strided_padded_cfft1d extension not available",
)


TYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex32": torch.complex32,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}

# Only get configs if strided_padded_cfft1d is available
if zipfft.strided_padded_cfft1d is not None:
    ALL_CONFIGS = zipfft.strided_padded_cfft1d.get_supported_configs()
    FORWARD_FFT_CONFIGS = [
        (cfg[0], cfg[1], cfg[2], cfg[3]) for cfg in ALL_CONFIGS if cfg[4] is True
    ]
    INVERSE_FFT_CONFIGS = [
        (cfg[0], cfg[1], cfg[2], cfg[3]) for cfg in ALL_CONFIGS if cfg[4] is False
    ]
else:
    FORWARD_FFT_CONFIGS = []
    INVERSE_FFT_CONFIGS = []
DATA_TYPES = [torch.complex64]


def run_forward_strided_padded_fft_test(
    fft_size: int,
    signal_length: int,
    stride: int,
    batch_size: int = 1,
    dtype: torch.dtype = torch.complex64,
):
    """Runs a single forward strided padded FFT test for given parameters.

    Parameters
    ----------
    fft_size : int
        The total size of the FFT (including padding).
    signal_length : int
        The actual length of the signal (without padding).
    stride : int
        The stride size (second dimension).
    batch_size : int, optional
        The batch size, by default 1.
    dtype : torch.dtype, optional
        The data type of the input tensor, by default torch.complex64.
    """
    # Create input tensor with appropriate shape (unpadded)
    input_shape = (signal_length, stride)
    output_shape = (fft_size, stride)
    if batch_size > 1:
        input_shape = (batch_size,) + input_shape
        output_shape = (batch_size,) + output_shape

    # Create input and output tensors
    input_tensor = torch.randn(input_shape, dtype=dtype, device="cuda")
    output_tensor = torch.zeros(output_shape, dtype=dtype, device="cuda")
    
    print(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
    print(f"Output tensor shape: {output_tensor.shape}, dtype: {output_tensor.dtype}")

    # PyTorch reference: Run the FFT along the first dimension with n=fft_size
    expected = torch.fft.fft(input_tensor.clone(), n=fft_size, dim=-2)

    # Run the zipFFT implementation
    zipfft.strided_padded_cfft1d.psfft(input_tensor, output_tensor, fft_size)

    assert torch.allclose(
        expected, output_tensor, atol=1e-4
    ), "Strided padded FFT results do not match ground truth"


def run_inverse_strided_padded_fft_test(
    fft_size: int,
    signal_length: int,
    stride: int,
    batch_size: int = 1,
    dtype: torch.dtype = torch.complex64,
):
    """Runs a single inverse strided padded FFT test for given parameters.

    Parameters
    ----------
    fft_size : int
        The total size of the FFT (including padding).
    signal_length : int
        The actual length of the signal (without padding).
    stride : int
        The stride size (second dimension).
    batch_size : int, optional
        The batch size, by default 1.
    dtype : torch.dtype, optional
        The data type of the input tensor, by default torch.complex64.
    """
    # Create input tensor with appropriate shape
    input_shape = (fft_size, stride)
    output_shape = (signal_length, stride)
    if batch_size > 1:
        input_shape = (batch_size,) + input_shape
        output_shape = (batch_size,) + output_shape

    # Create input and output tensors
    input_tensor = torch.randn(input_shape, dtype=dtype, device="cuda")
    output_tensor = torch.zeros(output_shape, dtype=dtype, device="cuda")

    # PyTorch reference implementation
    expected = torch.fft.ifft(input_tensor, dim=-2, norm="forward")
    expected = expected[..., :signal_length, :]

    # Run the zipFFT implementation
    zipfft.strided_padded_cfft1d.psifft(input_tensor, output_tensor, fft_size)

    assert torch.allclose(
        expected, output_tensor, atol=1e-4
    ), "Strided padded IFFT results do not match ground truth"


@pytest.mark.parametrize(
    "fft_size,signal_length,stride,batch_size", FORWARD_FFT_CONFIGS
)
@pytest.mark.parametrize("dtype", DATA_TYPES)
def test_strided_padded_fft_c2c_1d(fft_size, signal_length, stride, batch_size, dtype):
    """Test forward strided padded FFT for specific size, signal length, stride, batch size, and dtype."""
    run_forward_strided_padded_fft_test(
        fft_size=fft_size,
        signal_length=signal_length,
        stride=stride,
        batch_size=batch_size,
        dtype=dtype,
    )


@pytest.mark.parametrize(
    "fft_size,signal_length,stride,batch_size", INVERSE_FFT_CONFIGS
)
@pytest.mark.parametrize("dtype", DATA_TYPES)
def test_strided_padded_ifft_c2c_1d(fft_size, signal_length, stride, batch_size, dtype):
    """Test inverse strided padded FFT for specific size, signal length, stride, batch size, and dtype."""
    run_inverse_strided_padded_fft_test(
        fft_size=fft_size,
        signal_length=signal_length,
        stride=stride,
        batch_size=batch_size,
        dtype=dtype,
    )
