"""Real 2D convolution tests using PyTorch as a reference implementation."""

import torch  # !!! NOTE !!! CUDA backend built by PyTorch needs torch imported first!
import zipfft

import pytest

# Skip entire module if rfft2d is not available
pytestmark = pytest.mark.skipif(
    not zipfft.is_extension_available("padded_rconv2d"),
    reason="padded_rconv2d extension not available",
)

TYPE_MAP = {
    "float32": torch.float32,
    "complex64": torch.complex64,
}

# Only get configs if padded_rconv2d is available
# NOTE: Each element in the configuration tuple is as follows:
# (signal_length_y, signal_length_x, fft_size_y, fft_size_x, batch, cross_correlate)
if zipfft.padded_rconv2d is not None:
    ALL_CONFIGS = zipfft.padded_rconv2d.get_supported_conv_configs()

    CONV_CONFIGS = [
        (cfg[0], cfg[1], cfg[2], cfg[3], cfg[4])
        for cfg in ALL_CONFIGS
        if cfg[5] is False
    ]
    CROSS_CORR_CONFIGS = [
        (cfg[0], cfg[1], cfg[2], cfg[3], cfg[4])
        for cfg in ALL_CONFIGS
        if cfg[5] is True
    ]

NUM_TEST_REPEATS = 10
RTOL = 1e0  # NOTE: Effectively no rtol since numerical differences affect small values
ATOL = 5e-5


def run_convolution_2d_test(
    signal_length_y: int,
    signal_length_x: int,
    fft_size_y: int,
    fft_size_x: int,
    batch_size: int,
    dtype: torch.dtype = torch.float32,
    rtol: float = RTOL,
    atol: float = ATOL,
):
    """Runs a single real 2D convolution test for given parameters.

    Parameters
    ----------
    signal_length_y : int
        Length of the signal along the Y dimension (height).
    signal_length_x : int
        Length of the signal along the X dimension (width).
    fft_size_y : int
        Size of the FFT along the Y dimension.
    fft_size_x : int
        Size of the FFT along the X dimension.
    batch_size : int
        The batch size.
    dtype : torch.dtype
        The data type of the input tensor.
    rtol : float
        Absolute tolerance for comparison.
    atol : float
        Relative tolerance for comparison.
    """
    filter_shape = (signal_length_y, signal_length_x)
    image_shape = (fft_size_y, fft_size_x)
    output_shape = (fft_size_y - signal_length_y + 1, fft_size_x - signal_length_x + 1)
    if batch_size > 1:
        filter_shape = (batch_size,) + filter_shape
        output_shape = (batch_size,) + output_shape

    # Create a random input image and pre-transform
    input_image = torch.randn(image_shape, dtype=dtype, device="cuda")
    input_image_fft = torch.fft.rfftn(input_image)

    # Create a random filter to convolve with the input image
    input_filter = torch.randn(filter_shape, dtype=dtype, device="cuda")

    # Create the FFT workspace and output cross-correlogram tensors
    conv_workspace = torch.empty(
        batch_size,
        fft_size_y,
        fft_size_x // 2 + 1,
        dtype=torch.complex64,
        device="cuda",
    )
    output_cross_corr = torch.empty(output_shape, dtype=dtype, device="cuda")

    # PyTorch reference: Doing a FFT-based convolution
    filter_fft = torch.fft.rfftn(input_filter, s=(fft_size_y, fft_size_x), dim=(-2, -1))
    torch_conv_result_fft = input_image_fft[None, ...] * filter_fft
    torch_conv_result = torch.fft.irfftn(
        torch_conv_result_fft, dim=(-2, -1), norm="backward"
    )
    torch_conv_result = torch_conv_result[..., : output_shape[-2], : output_shape[-1]]

    # Run our implementation
    zipfft.padded_rconv2d.conv(
        input_filter,
        conv_workspace,
        input_image_fft,
        output_cross_corr,
        fft_size_y,
        fft_size_x,
    )
    output_cross_corr /= fft_size_y * fft_size_x

    # Verify results
    max_abs_diff = torch.max(torch.abs(torch_conv_result - output_cross_corr))
    max_rel_diff = torch.max(
        torch.abs(torch_conv_result - output_cross_corr)
        / (torch.abs(output_cross_corr) + 1e-8)
    )
    error_msg = (
        f"Real 2D convolution results do not match ground truth. "
        f"Max abs diff: {max_abs_diff}, Max rel diff: {max_rel_diff}. "
        f"Min/Max ground truth: {torch.min(torch_conv_result.abs())}, {torch.max(torch_conv_result.abs())}"
    )
    assert torch.allclose(torch_conv_result, output_cross_corr, rtol=rtol), error_msg


def run_cross_correlation_2d_test(
    signal_length_y: int,
    signal_length_x: int,
    fft_size_y: int,
    fft_size_x: int,
    batch_size: int,
    dtype: torch.dtype = torch.float32,
    rtol: float = RTOL,
    atol: float = ATOL,
):
    """Runs a single real 2D cross-correlation test for given parameters.

    Parameters
    ----------
    signal_length_y : int
        Length of the signal along the Y dimension (height).
    signal_length_x : int
        Length of the signal along the X dimension (width).
    fft_size_y : int
        Size of the FFT along the Y dimension.
    fft_size_x : int
        Size of the FFT along the X dimension.
    batch_size : int
        The batch size.
    dtype : torch.dtype
        The data type of the input tensor.
    rtol : float
        Absolute tolerance for comparison.
    atol : float
        Relative tolerance for comparison.
    """
    filter_shape = (signal_length_y, signal_length_x)
    image_shape = (fft_size_y, fft_size_x)
    output_shape = (fft_size_y - signal_length_y + 1, fft_size_x - signal_length_x + 1)
    if batch_size > 1:
        filter_shape = (batch_size,) + filter_shape
        output_shape = (batch_size,) + output_shape

    # Create a random input image and pre-transform
    input_image = torch.randn(image_shape, dtype=dtype, device="cuda")
    input_image_fft = torch.fft.rfftn(input_image)

    # Create a random filter to cross-correlate with the input image
    input_filter = torch.randn(filter_shape, dtype=dtype, device="cuda")

    # Create the FFT workspace and output cross-correlogram tensors
    corr_workspace = torch.empty(
        batch_size,
        fft_size_y,
        fft_size_x // 2 + 1,
        dtype=torch.complex64,
        device="cuda",
    )
    output_cross_corr = torch.empty(output_shape, dtype=dtype, device="cuda")

    # PyTorch reference: Doing a FFT-based cross-correlation
    filter_fft = torch.fft.rfftn(input_filter, s=(fft_size_y, fft_size_x), dim=(-2, -1))
    torch_corr_result_fft = input_image_fft[None, ...] * torch.conj(filter_fft)
    torch_corr_result = torch.fft.irfftn(
        torch_corr_result_fft, dim=(-2, -1), norm="backward"
    )
    torch_corr_result = torch_corr_result[..., : output_shape[-2], : output_shape[-1]]

    # Run our implementation
    zipfft.padded_rconv2d.corr(
        input_filter,
        corr_workspace,
        input_image_fft,
        output_cross_corr,
        fft_size_y,
        fft_size_x,
    )
    output_cross_corr /= fft_size_y * fft_size_x

    # Verify results
    max_abs_diff = torch.max(torch.abs(torch_corr_result - output_cross_corr))
    max_rel_diff = torch.max(
        torch.abs(torch_corr_result - output_cross_corr)
        / (torch.abs(output_cross_corr) + 1e-8)
    )
    error_msg = (
        f"Real 2D cross-correlation results do not match ground truth. "
        f"Max abs diff: {max_abs_diff}, Max rel diff: {max_rel_diff}. "
        f"Min/Max ground truth: {torch.min(torch_corr_result.abs())}, {torch.max(torch_corr_result.abs())}"
    )
    assert torch.allclose(
        torch_corr_result, output_cross_corr, rtol=rtol, atol=atol
    ), error_msg


@pytest.mark.parametrize(
    "signal_length_y,signal_length_x,fft_size_y,fft_size_x,batch_size",
    CONV_CONFIGS,
)
def test_convolution_2d(
    signal_length_y: int,
    signal_length_x: int,
    fft_size_y: int,
    fft_size_x: int,
    batch_size: int,
):
    """Tests real 2D convolution for given parameters.

    Parameters
    ----------
    signal_length_y : int
        Length of the signal along the Y dimension (height).
    signal_length_x : int
        Length of the signal along the X dimension (width).
    fft_size_y : int
        Size of the FFT along the Y dimension.
    fft_size_x : int
        Size of the FFT along the X dimension.
    batch_size : int
        The batch size.
    """
    for _ in range(NUM_TEST_REPEATS):
        run_convolution_2d_test(
            signal_length_y,
            signal_length_x,
            fft_size_y,
            fft_size_x,
            batch_size,
            dtype=torch.float32,
        )


@pytest.mark.parametrize(
    "signal_length_y,signal_length_x,fft_size_y,fft_size_x,batch_size",
    CROSS_CORR_CONFIGS,
)
def test_cross_correlation_2d(
    signal_length_y: int,
    signal_length_x: int,
    fft_size_y: int,
    fft_size_x: int,
    batch_size: int,
):
    """Tests real 2D cross-correlation for given parameters.

    Parameters
    ----------
    signal_length_y : int
        Length of the signal along the Y dimension (height).
    signal_length_x : int
        Length of the signal along the X dimension (width).
    fft_size_y : int
        Size of the FFT along the Y dimension.
    fft_size_x : int
        Size of the FFT along the X dimension.
    batch_size : int
        The batch size.
    """
    for _ in range(NUM_TEST_REPEATS):
        run_cross_correlation_2d_test(
            signal_length_y,
            signal_length_x,
            fft_size_y,
            fft_size_x,
            batch_size,
            dtype=torch.float32,
        )
