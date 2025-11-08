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
#  transpose_axes, conv_data_is_transposed)
if zipfft.padded_rconv2d is not None:
    ALL_CONFIGS = zipfft.padded_rconv2d.get_supported_conv_configs()

    CONV_CONFIGS = [
        (cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[6], cfg[7])
        for cfg in ALL_CONFIGS
        if cfg[5] is False
    ]
    CROSS_CORR_CONFIGS = [
        (cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[6], cfg[7])
        for cfg in ALL_CONFIGS
        if cfg[5] is True
    ]

NUM_TEST_REPEATS = 10
RTOL = 3e0  # NOTE: Effectively no rtol since numerical differences affect small values
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
    transpose_axes: bool = False,
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
    transpose_axes : bool
        Whether to transpose the FFT workspace internally.
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

    # Create a random input image and pre-transform
    input_image = torch.randn(image_shape, dtype=dtype, device="cuda")
    input_image_fft = torch.fft.rfftn(input_image)

    # # Manually set the fft of the input image for debugging
    # for i in range(input_image_fft.shape[0]):
    #     for j in range(input_image_fft.shape[1]):
    #         input_image_fft[i, j] = complex(i, j)

    # Create a random filter to convolve/correlate with the input image
    input_filter = torch.randn(filter_shape, dtype=dtype, device="cuda")

    # total = 1
    # for dim in filter_shape:
    #     total *= dim
    # input_filter = torch.arange(total, dtype=dtype, device="cuda").reshape(filter_shape)

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
    if conv_data_is_transposed:
        # Transpose from (fft_size_y, fft_size_x//2+1) to (fft_size_x//2+1, fft_size_y)
        zipfft_input_image_fft = input_image_fft.T.contiguous()

    # Create the FFT workspace and output tensors
    workspace_shape = (batch_size, fft_size_y, fft_size_x // 2 + 1)
    workspace = torch.empty(
        *workspace_shape,
        dtype=torch.complex64,
        device="cuda",
    )
    if transpose_axes:
        tmp = torch.empty(
            workspace_shape[0],
            workspace_shape[2],
            workspace_shape[1],
            dtype=torch.complex64,
            device="cuda",
        )
        tmp.copy_(workspace.permute(0, 2, 1))
        workspace = tmp
        assert workspace.is_contiguous()
    if batch_size == 1:
        workspace = workspace.squeeze(0)

    # Run our implementation
    if cross_correlate:
        zipfft.padded_rconv2d.corr(
            input_filter,
            workspace,
            zipfft_input_image_fft,
            output,
            fft_size_y,
            fft_size_x,
            transpose_axes,
            conv_data_is_transposed,
        )
    else:
        zipfft.padded_rconv2d.conv(
            input_filter,
            workspace,
            zipfft_input_image_fft,
            output,
            fft_size_y,
            fft_size_x,
            transpose_axes,
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
        f"Config: transpose_axes={transpose_axes}, conv_data_is_transposed={conv_data_is_transposed}"
    )

    # ### DEBUGGING: Save the tensors if the test fails for further inspection
    # if not torch.allclose(torch_result, output, rtol=rtol, atol=atol):
    #     import numpy as np

    #     torch_result_np = torch_result.cpu().numpy()
    #     zipfft_result_np = output.cpu().numpy()
    #     np.save(f"torch_{op_name}_result.npy", torch_result_np)
    #     np.save(f"zipfft_{op_name}_result.npy", zipfft_result_np)
    # ### END DEBUGGING

    assert torch.allclose(torch_result, output, rtol=rtol, atol=atol), error_msg
    # assert False, "DEBUGGING"


@pytest.mark.parametrize(
    "signal_length_y,signal_length_x,fft_size_y,fft_size_x,batch_size,transpose_axes,conv_data_is_transposed",
    CONV_CONFIGS,
)
def test_convolution_2d(
    signal_length_y: int,
    signal_length_x: int,
    fft_size_y: int,
    fft_size_x: int,
    batch_size: int,
    transpose_axes: bool,
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
    transpose_axes : bool
        Whether to transpose the FFT workspace internally.
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
            transpose_axes=transpose_axes,
            conv_data_is_transposed=conv_data_is_transposed,
            dtype=torch.float32,
        )


@pytest.mark.parametrize(
    "signal_length_y,signal_length_x,fft_size_y,fft_size_x,batch_size,transpose_axes,conv_data_is_transposed",
    CROSS_CORR_CONFIGS,
)
def test_cross_correlation_2d(
    signal_length_y: int,
    signal_length_x: int,
    fft_size_y: int,
    fft_size_x: int,
    batch_size: int,
    transpose_axes: bool,
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
    transpose_axes : bool
        Whether to transpose the FFT workspace internally.
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
            transpose_axes=transpose_axes,
            conv_data_is_transposed=conv_data_is_transposed,
            dtype=torch.float32,
        )
