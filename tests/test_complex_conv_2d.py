"""Complex-to-complex 2D convolution tests using PyTorch as a reference implementation."""

import torch  # !!! NOTE !!! CUDA backend built by PyTorch needs torch imported first!
import zipfft

import pytest

# Skip entire module if padded_cconv2d is not available
pytestmark = pytest.mark.skipif(
    not zipfft.is_extension_available("padded_cconv2d"),
    reason="padded_cconv2d extension not available",
)

FFT_NORMALIZATION_DIRECTION = "backward"  # Options: 'forward', 'backward', 'ortho'

# Only get configs if padded_cconv2d is available
# NOTE: Each element in the configuration tuple is as follows:
# (signal_length_y, signal_length_x, fft_size_y, fft_size_x, batch, cross_correlate)
if zipfft.padded_cconv2d is not None:
    ALL_CONFIGS = zipfft.padded_cconv2d.get_supported_conv_configs()

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
RTOL = 1e-3
ATOL = 5e-6


def run_complex_conv_or_corr_2d_test(
    signal_length_y: int,
    signal_length_x: int,
    fft_size_y: int,
    fft_size_x: int,
    batch_size: int,
    cross_correlate: bool,
    dtype: torch.dtype = torch.complex64,
    rtol: float = RTOL,
    atol: float = ATOL,
    device: str | torch.device = "cuda:0",
):
    """Runs a single complex-to-complex 2D convolution or cross-correlation test.

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
    dtype : torch.dtype
        The data type of the input tensor (must be a complex dtype).
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
    StrideY = fft_size_x

    # Create a random complex input image and pre-transform
    input_image = torch.complex(
        torch.randn(image_shape, dtype=torch.float32, device=device),
        torch.randn(image_shape, dtype=torch.float32, device=device),
    ).to(dtype)
    input_image_fft = torch.fft.fft2(input_image).contiguous()

    # Create a random complex filter to convolve/correlate with the input image
    input_filter = torch.complex(
        torch.randn(filter_shape, dtype=torch.float32, device=device),
        torch.randn(filter_shape, dtype=torch.float32, device=device),
    ).to(dtype)

    # Create the FFT workspace tensor
    # Shape: (batch, fft_size_y, fft_size_x) matching the binding expectations
    if batch_size > 1:
        workspace_shape = (batch_size, fft_size_y, StrideY)
    else:
        workspace_shape = (fft_size_y, StrideY)

    fft_workspace = torch.empty(
        workspace_shape,
        dtype=torch.complex64,
        device=device,
    )

    output = torch.empty(output_shape, dtype=dtype, device=device)

    # PyTorch reference: Doing a FFT-based convolution or cross-correlation
    filter_fft = torch.fft.fft2(input_filter, s=(fft_size_y, fft_size_x), dim=(-2, -1))

    if cross_correlate:
        # Cross-correlation: conjugate the filter
        torch_result_fft = input_image_fft[None, ...] * torch.conj(filter_fft)
    else:
        # Convolution: no conjugate
        torch_result_fft = input_image_fft[None, ...] * filter_fft

    torch_result = torch.fft.ifft2(
        torch_result_fft, dim=(-2, -1), norm=FFT_NORMALIZATION_DIRECTION
    )
    torch_result = torch_result[..., : output_shape[-2], : output_shape[-1]]

    # Transpose the 'input_image_fft' along last two dimensions into contiguous layout
    # with shape (..., fft_size_x, fft_size_y)
    input_image_fft = input_image_fft.transpose(-2, -1).contiguous()

    # Run our implementation
    if cross_correlate:
        zipfft.padded_cconv2d.corr(
            input_filter,
            fft_workspace,
            input_image_fft,
            output,
            fft_size_y,
            fft_size_x,
        )
    else:
        zipfft.padded_cconv2d.conv(
            input_filter,
            fft_workspace,
            input_image_fft,
            output,
            fft_size_y,
            fft_size_x,
        )

    # Synchronize to ensure all operations are complete
    torch.cuda.synchronize()

    # Verify results
    max_abs_diff = torch.max(torch.abs(torch_result - output))
    max_rel_diff = torch.max(
        torch.abs(torch_result - output) / (torch.abs(output) + 1e-8)
    )

    op_name = "cross-correlation" if cross_correlate else "convolution"
    error_msg = (
        f"Complex 2D {op_name} results do not match ground truth. "
        f"Max abs diff: {max_abs_diff}, Max rel diff: {max_rel_diff}. "
        f"Min/Max ground truth: "
        f"{torch.min(torch_result.abs())}, {torch.max(torch_result.abs())}."
    )

    # For small sizes (fft_size <= 512) use allclose check, but for larger transforms,
    # check the L2 norm instead to avoid failures due to implementation differences
    if max(fft_size_y, fft_size_x) < 512:
        assert torch.allclose(torch_result, output, rtol=rtol, atol=atol), error_msg
    else:
        l2_norm = torch.norm(torch_result - output) / torch.norm(torch_result)
        assert l2_norm < atol, error_msg + f" L2 norm: {l2_norm}"


@pytest.mark.parametrize(
    "signal_length_y,signal_length_x,fft_size_y,fft_size_x,batch_size",
    CONV_CONFIGS,
)
def test_complex_convolution_2d(
    signal_length_y: int,
    signal_length_x: int,
    fft_size_y: int,
    fft_size_x: int,
    batch_size: int,
):
    """Tests complex-to-complex 2D convolution for given parameters.

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
        run_complex_conv_or_corr_2d_test(
            signal_length_y,
            signal_length_x,
            fft_size_y,
            fft_size_x,
            batch_size,
            cross_correlate=False,
            dtype=torch.complex64,
        )


@pytest.mark.parametrize(
    "signal_length_y,signal_length_x,fft_size_y,fft_size_x,batch_size",
    CROSS_CORR_CONFIGS,
)
def test_complex_cross_correlation_2d(
    signal_length_y: int,
    signal_length_x: int,
    fft_size_y: int,
    fft_size_x: int,
    batch_size: int,
):
    """Tests complex-to-complex 2D cross-correlation for given parameters.

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
        run_complex_conv_or_corr_2d_test(
            signal_length_y,
            signal_length_x,
            fft_size_y,
            fft_size_x,
            batch_size,
            cross_correlate=True,
            dtype=torch.complex64,
        )


def test_complex_non_standard_stream():
    """Tests execution on the non-default CUDA stream"""
    stream = torch.cuda.Stream()

    # Get the params for the zeroth convolution config
    (
        signal_length_y,
        signal_length_x,
        fft_size_y,
        fft_size_x,
        batch_size,
    ) = CONV_CONFIGS[0]

    for _ in range(NUM_TEST_REPEATS):
        with torch.cuda.stream(stream):
            run_complex_conv_or_corr_2d_test(
                signal_length_y,
                signal_length_x,
                fft_size_y,
                fft_size_x,
                batch_size,
                cross_correlate=False,
                dtype=torch.complex64,
            )

    # Synchronize to ensure all operations are complete
    stream.synchronize()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Less than 2 CUDA devices available",
)
def test_complex_non_standard_device():
    """Tests execution of a single configuration on the non-zero device"""
    device = "cuda:1"

    # Get the params for the zeroth cross-correlation config
    (
        signal_length_y,
        signal_length_x,
        fft_size_y,
        fft_size_x,
        batch_size,
    ) = CROSS_CORR_CONFIGS[0]

    for _ in range(NUM_TEST_REPEATS):
        run_complex_conv_or_corr_2d_test(
            signal_length_y,
            signal_length_x,
            fft_size_y,
            fft_size_x,
            batch_size,
            cross_correlate=True,
            dtype=torch.complex64,
            device=device,
        )
