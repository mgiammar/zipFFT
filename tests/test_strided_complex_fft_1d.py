"""Strided complex-to-complex 1D FFT tests for cuFFTDx comparing against PyTorch."""

import torch  # !!! NOTE !!! CUDA backend built by PyTorch needs torch imported first!
import zipfft

import pytest

# Skip entire module if strided_cfft1d is not available
pytestmark = pytest.mark.skipif(
    not zipfft.is_extension_available("strided_cfft1d"),
    reason="strided_cfft1d extension not available",
)


TYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex32": torch.complex32,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}

# Only get configs if strided_cfft1d is available
if zipfft.strided_cfft1d is not None:
    ALL_CONFIGS = zipfft.strided_cfft1d.get_supported_fft_configs()
    FORWARD_FFT_CONFIGS = [
        (cfg[0], cfg[1], cfg[2]) for cfg in ALL_CONFIGS if cfg[3] is True
    ]
    INVERSE_FFT_CONFIGS = [
        (cfg[0], cfg[1], cfg[2]) for cfg in ALL_CONFIGS if cfg[3] is False
    ]
else:
    FORWARD_FFT_CONFIGS = []
    INVERSE_FFT_CONFIGS = []
DATA_TYPES = [torch.complex64]


def run_forward_strided_fft_test(
    fft_size: int,
    stride: int,
    batch_size: int = 1,
    dtype: torch.dtype = torch.complex64,
):
    """Runs a single forward strided FFT test for given parameters.

    Parameters
    ----------
    fft_size : int
        The size of the FFT to run along the first dimension.
    stride : int
        The stride size (second dimension).
    batch_size : int, optional
        The batch size, by default 1.
    dtype : torch.dtype, optional
        The data type of the input tensor, by default torch.complex64.
    """
    # Create input tensor with appropriate shape
    if batch_size == 1:
        shape = (fft_size, stride)
    else:
        shape = (batch_size, fft_size, stride)

    x0 = torch.randn(shape, dtype=dtype, device="cuda")
    x1 = x0.clone()

    # PyTorch reference: FFT along the first non-batch dimension (dim=-2 for strided)
    if batch_size == 1:
        torch.fft.fft(x0, dim=0, out=x0)
    else:
        torch.fft.fft(x0, dim=1, out=x0)

    # NOTE: This zipFFT function is in-place
    zipfft.strided_cfft1d.fft(x1)

    assert torch.allclose(
        x0, x1, atol=1e-4
    ), "Strided FFT results do not match ground truth"


def run_inverse_strided_fft_test(
    fft_size: int,
    stride: int,
    batch_size: int = 1,
    dtype: torch.dtype = torch.complex64,
):
    """Runs a single inverse strided FFT test for given parameters.

    Parameters
    ----------
    fft_size : int
        The size of the FFT to run along the first dimension.
    stride : int
        The stride size (second dimension).
    batch_size : int, optional
        The batch size, by default 1.
    dtype : torch.dtype, optional
        The data type of the input tensor, by default torch.complex64.
    """
    # Create input tensor with appropriate shape
    if batch_size == 1:
        shape = (fft_size, stride)
    else:
        shape = (batch_size, fft_size, stride)

    x0 = torch.randn(shape, dtype=dtype, device="cuda")
    x1 = x0.clone()

    # PyTorch reference: IFFT along the first non-batch dimension (dim=-2 for strided)
    if batch_size == 1:
        torch.fft.ifft(x0, dim=0, out=x0)
        x0 *= float(fft_size)  # Scale to match cuFFTDx inverse FFT definition
    else:
        torch.fft.ifft(x0, dim=1, out=x0)
        x0 *= float(fft_size)  # Scale to match cuFFTDx inverse FFT definition

    # NOTE: This zipFFT function is in-place
    zipfft.strided_cfft1d.ifft(x1)

    assert torch.allclose(
        x0, x1, atol=1e-4
    ), "Strided IFFT results do not match ground truth"


@pytest.mark.parametrize("fft_size,stride,batch_size", FORWARD_FFT_CONFIGS)
@pytest.mark.parametrize("dtype", DATA_TYPES)
def test_strided_fft_c2c_1d(fft_size, stride, batch_size, dtype):
    """Test forward strided FFT for specific size, stride, batch size, and dtype."""
    run_forward_strided_fft_test(
        fft_size=fft_size, stride=stride, batch_size=batch_size, dtype=dtype
    )


@pytest.mark.parametrize("fft_size,stride,batch_size", INVERSE_FFT_CONFIGS)
@pytest.mark.parametrize("dtype", DATA_TYPES)
def test_strided_ifft_c2c_1d(fft_size, stride, batch_size, dtype):
    """Test inverse strided FFT for specific size, stride, batch size, and dtype."""
    run_inverse_strided_fft_test(
        fft_size=fft_size, stride=stride, batch_size=batch_size, dtype=dtype
    )
