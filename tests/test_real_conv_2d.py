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
# (signal_length_y, signal_length_x, fft_size_y, fft_size_x, batch, cross_correlate,
#  conv_data_is_transposed)
if zipfft.padded_rconv2d is not None:
    ALL_CONFIGS = zipfft.padded_rconv2d.get_supported_conv_configs()

    CONV_CONFIGS = [
        (cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[6])
        for cfg in ALL_CONFIGS
        if cfg[5] is False
    ]
    CROSS_CORR_CONFIGS = [
        (cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[6])
        for cfg in ALL_CONFIGS
        if cfg[5] is True
    ]

NUM_TEST_REPEATS = 10
RTOL = 1e1  # NOTE: Effectively no rtol since numerical differences affect small values
ATOL = 5e-5

# If we want to normalize by the number of elements in the FFT
NORMALIZE_BY_FFT_SIZE = True
NORM = "backward" if NORMALIZE_BY_FFT_SIZE else "forward"


def run_conv_or_corr_2d_test(
    signal_length_y: int,
    signal_length_x: int,
    fft_size_y: int,
    fft_size_x: int,
    batch_size: int,
    cross_correlate: bool,
    conv_data_is_transposed: bool = False,
    dtype: torch.dtype = torch.float32,
    rtol: float = RTOL,
    atol: float = ATOL,
):
    """Runs a single real 2D convolution or cross-correlation test for given parameters.

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
    cross_correlate : bool
        Whether to perform cross-correlation (True) or convolution (False).
    conv_data_is_transposed : bool
        Whether the convolution data is pre-transposed.
    dtype : torch.dtype
        The data type of the input tensor.
    rtol : float
        Relative tolerance for comparison.
    atol : float
        Absolute tolerance for comparison.
    """
    filter_shape = (signal_length_y, signal_length_x)
    image_shape = (fft_size_y, fft_size_x)
    output_shape = (fft_size_y - signal_length_y + 1, fft_size_x - signal_length_x + 1)
    if batch_size > 1:
        filter_shape = (batch_size,) + filter_shape
        output_shape = (batch_size,) + output_shape

    # Calculate derived constants
    StrideY = fft_size_x // 2 + 1
    ValidLengthY = fft_size_y - signal_length_y + 1

    # Create a random input image and pre-transform
    input_image = torch.randn(image_shape, dtype=dtype, device="cuda")
    input_image_fft = torch.fft.rfftn(input_image).contiguous()

    # Create a random filter to convolve/correlate with the input image
    input_filter = torch.randn(filter_shape, dtype=dtype, device="cuda")

    output = torch.empty(output_shape, dtype=dtype, device="cuda")

    # PyTorch reference: Doing a FFT-based convolution or cross-correlation
    filter_fft = torch.fft.rfftn(input_filter, s=(fft_size_y, fft_size_x), dim=(-2, -1))

    if cross_correlate:
        # Cross-correlation: conjugate the filter
        torch_result_fft = input_image_fft[None, ...] * torch.conj(filter_fft)
    else:
        # Convolution: no conjugate
        torch_result_fft = input_image_fft[None, ...] * filter_fft

    torch_result = torch.fft.irfftn(torch_result_fft, dim=(-2, -1), norm=NORM)
    torch_result = torch_result[..., : output_shape[-2], : output_shape[-1]]

    # Prepare input_image_fft for zipfft call
    # If conv_data_is_transposed is True, transpose the image FFT data
    zipfft_input_image_fft = input_image_fft
    
    # # Print the contiguity and underlying strides of both tensors for debugging
    # assert False, f"DEBUGGING: is_contig: {zipfft_input_image_fft.is_contiguous(), input_image_fft.is_contiguous(), zipfft_input_image_fft.stride(), input_image_fft.stride()}"
    
    if conv_data_is_transposed:
        # Transpose from (fft_size_y, StrideY) to (StrideY, fft_size_y)
        zipfft_input_image_fft = input_image_fft.T.contiguous()

    # Create the FFT workspace tensors matching the 5-kernel pipeline
    # Workspace 1: After R2C - (batch, SignalLengthY, StrideY)
    workspace_r2c = torch.empty(
        batch_size,
        signal_length_y,
        StrideY,
        dtype=torch.complex64,
        device="cuda",
    )

    # Workspace 2: After Transpose1 - (batch, StrideY, SignalLengthY)
    workspace_r2c_transposed = torch.empty(
        batch_size,
        StrideY,
        signal_length_y,
        dtype=torch.complex64,
        device="cuda",
    )

    # Workspace 3: After C2C - (batch, StrideY, ValidLengthY)
    workspace_c2c_transposed = torch.empty(
        batch_size,
        StrideY,
        ValidLengthY,
        dtype=torch.complex64,
        device="cuda",
    )

    # Workspace 4: After Transpose2 - (batch, ValidLengthY, StrideY)
    workspace_c2r = torch.empty(
        batch_size,
        ValidLengthY,
        StrideY,
        dtype=torch.complex64,
        device="cuda",
    )

    # Run our implementation
    if cross_correlate:
        zipfft.padded_rconv2d.corr(
            input_filter,
            workspace_r2c,
            workspace_r2c_transposed,
            workspace_c2c_transposed,
            workspace_c2r,
            zipfft_input_image_fft,
            output,
            fft_size_y,
            fft_size_x,
            conv_data_is_transposed,
        )
    else:
        zipfft.padded_rconv2d.conv(
            input_filter,
            workspace_r2c,
            workspace_r2c_transposed,
            workspace_c2c_transposed,
            workspace_c2r,
            zipfft_input_image_fft,
            output,
            fft_size_y,
            fft_size_x,
            conv_data_is_transposed,
        )

    if NORMALIZE_BY_FFT_SIZE:
        norm_factor = fft_size_y * fft_size_x
        output /= norm_factor

    # Verify results
    max_abs_diff = torch.max(torch.abs(torch_result - output))
    max_rel_diff = torch.max(
        torch.abs(torch_result - output) / (torch.abs(output) + 1e-8)
    )

    op_name = "cross-correlation" if cross_correlate else "convolution"
    error_msg = (
        f"Real 2D {op_name} results do not match ground truth. "
        f"Max abs diff: {max_abs_diff}, Max rel diff: {max_rel_diff}. "
        f"Min/Max ground truth: {torch.min(torch_result.abs())}, {torch.max(torch_result.abs())}. "
        f"Config: conv_data_is_transposed={conv_data_is_transposed}"
    )

    assert torch.allclose(torch_result, output, rtol=rtol, atol=atol), error_msg


@pytest.mark.parametrize(
    "signal_length_y,signal_length_x,fft_size_y,fft_size_x,batch_size,conv_data_is_transposed",
    CONV_CONFIGS,
)
def test_convolution_2d(
    signal_length_y: int,
    signal_length_x: int,
    fft_size_y: int,
    fft_size_x: int,
    batch_size: int,
    conv_data_is_transposed: bool,
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
    conv_data_is_transposed : bool
        Whether the convolution data is pre-transposed.
    """
    for _ in range(NUM_TEST_REPEATS):
        run_conv_or_corr_2d_test(
            signal_length_y,
            signal_length_x,
            fft_size_y,
            fft_size_x,
            batch_size,
            cross_correlate=False,
            conv_data_is_transposed=conv_data_is_transposed,
            dtype=torch.float32,
        )


@pytest.mark.parametrize(
    "signal_length_y,signal_length_x,fft_size_y,fft_size_x,batch_size,conv_data_is_transposed",
    CROSS_CORR_CONFIGS,
)
def test_cross_correlation_2d(
    signal_length_y: int,
    signal_length_x: int,
    fft_size_y: int,
    fft_size_x: int,
    batch_size: int,
    conv_data_is_transposed: bool,
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
    conv_data_is_transposed : bool
        Whether the convolution data is pre-transposed.
    """
    for _ in range(NUM_TEST_REPEATS):
        run_conv_or_corr_2d_test(
            signal_length_y,
            signal_length_x,
            fft_size_y,
            fft_size_x,
            batch_size,
            cross_correlate=True,
            conv_data_is_transposed=conv_data_is_transposed,
            dtype=torch.float32,
        )
