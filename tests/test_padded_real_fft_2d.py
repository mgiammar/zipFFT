"""Padded real-to-complex and complex-to-real 2D FFT tests comparing against PyTorch."""

import torch  # !!! NOTE !!! CUDA backend built by PyTorch needs torch imported first!
import zipfft

import pytest

# Skip entire module if padded_rfft2d is not available
pytestmark = pytest.mark.skipif(
    not zipfft.is_extension_available("padded_rfft2d"),
    reason="padded_rfft2d extension not available",
)

TYPE_MAP = {
    "float32": torch.float32,
    "complex64": torch.complex64,
}

# Only get configs if padded_rfft2d is available
if zipfft.padded_rfft2d is not None:
    ALL_CONFIGS = zipfft.padded_rfft2d.get_supported_fft_configs()
    # Format: (signal_length_y, signal_length_x, fft_size_y, fft_size_x, batch_size, is_forward)
    FORWARD_FFT_CONFIGS = [
        (cfg[0], cfg[1], cfg[2], cfg[3], cfg[4])
        for cfg in ALL_CONFIGS
        if cfg[5] is True
    ]
    INVERSE_FFT_CONFIGS = [
        (cfg[0], cfg[1], cfg[2], cfg[3], cfg[4])
        for cfg in ALL_CONFIGS
        if cfg[5] is False
    ]
else:
    FORWARD_FFT_CONFIGS = []
    INVERSE_FFT_CONFIGS = []

# Currently only testing with float32/complex64
DATA_TYPES = [torch.float32]

# How many times to repeat each test to catch any non-deterministic issues
NUM_TEST_REPEATS = 10


def run_padded_forward_real_fft_2d_test(
    signal_length_y: int,  # actual signal height
    signal_length_x: int,  # actual signal width
    fft_size_y: int,  # padded fft height
    fft_size_x: int,  # padded fft width
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32,
    atol: float = 1e-3,  # Looser tolerance for accumulated multi-dimensional FFTs
):
    """Runs a single padded forward real 2D FFT test for given parameters.

    This test creates a smaller signal tensor, then performs zero-padding
    implicitly in our FFT and explicitly in PyTorch before running the FFT.
    """
    # Create input tensor with the signal shape
    input_shape = (signal_length_y, signal_length_x)
    stride_y = fft_size_x // 2 + 1
    output_shape = (fft_size_y, stride_y)
    if batch_size > 1:
        input_shape = (batch_size, signal_length_y, signal_length_x)
        output_shape = (batch_size, fft_size_y, stride_y)

    # Create random data for the actual signal (not padded yet)
    gen = torch.Generator(device="cuda")
    input_data = torch.randn(input_shape, dtype=dtype, device="cuda", generator=gen)

    # Create output tensor for our implementation
    output_data = torch.zeros(output_shape, dtype=torch.complex64, device="cuda")

    # PyTorch reference: rfft2
    torch_output = torch.fft.rfft2(input_data.clone(), s=(fft_size_y, fft_size_x))

    # Run our implementation with implicit padding
    zipfft.padded_rfft2d.fft(input_data, output_data, fft_size_y, fft_size_x)

    # Verify results
    assert torch.allclose(
        torch_output, output_data, atol=atol
    ), f"Padded real 2D FFT results do not match ground truth. Max diff: {torch.max(torch.abs(torch_output - output_data))}"


def run_padded_inverse_real_fft_2d_test(
    signal_length_y: int,  # actual signal height to recover
    signal_length_x: int,  # actual signal width to recover
    fft_size_y: int,  # padded fft height
    fft_size_x: int,  # padded fft width
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32,
    atol: float = 1e-3,  # Looser tolerance for accumulated multi-dimensional FFTs
):
    """Runs a single padded inverse real 2D FFT test for given parameters.

    This test starts with a complex tensor representing the FFT of a padded signal,
    then performs an inverse FFT to recover the original signal (truncating the padding).
    """
    # Create input complex tensor for the FFT result
    stride_y = fft_size_x // 2 + 1
    input_shape = (fft_size_y, stride_y)
    output_shape = (signal_length_y, signal_length_x)
    if batch_size > 1:
        input_shape = (batch_size, fft_size_y, stride_y)
        output_shape = (batch_size, signal_length_y, signal_length_x)

    # Start with random complex data (representing FFT coefficients)
    gen = torch.Generator(device="cuda")
    input_data = torch.randn(
        input_shape, dtype=torch.complex64, device="cuda", generator=gen
    )

    # Create output tensor for our implementation (unpadded)
    output_data = torch.zeros(output_shape, dtype=dtype, device="cuda")

    # PyTorch reference: irfft2 (will produce output that we need to truncate)
    torch_output = torch.fft.irfft2(input_data, norm="forward")
    torch_output = torch_output[..., :signal_length_y, :signal_length_x]

    # Run our implementation with implicit unpadding
    zipfft.padded_rfft2d.ifft(input_data, output_data, fft_size_y, fft_size_x)

    # Verify results
    assert torch.allclose(
        torch_output, output_data, atol=atol
    ), f"Padded inverse real 2D FFT results do not match ground truth. Max diff: {torch.max(torch.abs(torch_output - output_data))}"


@pytest.mark.parametrize(
    "signal_length_y,signal_length_x,fft_size_y,fft_size_x,batch_size",
    FORWARD_FFT_CONFIGS,
)
@pytest.mark.parametrize("dtype", DATA_TYPES)
def test_padded_real_fft_r2c_2d(
    signal_length_y,
    signal_length_x,
    fft_size_y,
    fft_size_x,
    batch_size,
    dtype,
):
    """Test padded forward real 2D FFT for specific sizes, strides, batch size, and dtype."""
    print(
        f"Running padded forward real 2D FFT test with input shape ({batch_size}, {signal_length_y}, {signal_length_x}) "
        f"and padded to ({batch_size}, {fft_size_y}, {fft_size_x}) with output shape ({batch_size}, {fft_size_y}, {fft_size_x // 2 + 1})"
    )

    for i in range(NUM_TEST_REPEATS):
        run_padded_forward_real_fft_2d_test(
            signal_length_y=signal_length_y,
            signal_length_x=signal_length_x,
            fft_size_y=fft_size_y,
            fft_size_x=fft_size_x,
            batch_size=batch_size,
            dtype=dtype,
        )


@pytest.mark.parametrize(
    "signal_length_y,signal_length_x,fft_size_y,fft_size_x,batch_size",
    INVERSE_FFT_CONFIGS,
)
@pytest.mark.parametrize("dtype", DATA_TYPES)
def test_padded_real_fft_c2r_2d(
    signal_length_y,
    signal_length_x,
    fft_size_y,
    fft_size_x,
    batch_size,
    dtype,
):
    """Test padded inverse real 2D FFT for specific sizes, strides, batch size, and dtype."""
    print(
        f"Running padded inverse real 2D FFT test with input shape ({batch_size}, {fft_size_y}, {fft_size_x // 2 + 1}) "
        f"and output shape ({batch_size}, {signal_length_y}, {signal_length_x})"
    )

    for i in range(NUM_TEST_REPEATS):
        run_padded_inverse_real_fft_2d_test(
            signal_length_y=signal_length_y,
            signal_length_x=signal_length_x,
            fft_size_y=fft_size_y,
            fft_size_x=fft_size_x,
            batch_size=batch_size,
            dtype=dtype,
        )


# Additional test to verify end-to-end FFT and inverse FFT
def test_padded_real_fft_round_trip():
    """Test that running a padded forward FFT followed by a padded inverse FFT
    recovers the original signal (within numerical precision)."""

    # Use medium-sized test case with batch size
    signal_length_y, signal_length_x = 96, 96
    fft_size_y, fft_size_x = 128, 128
    stride_y = fft_size_x // 2 + 1  # 65
    batch_size = 8

    # Create input tensor with random data
    input_shape = (batch_size, signal_length_y, signal_length_x)
    input_data = torch.randn(input_shape, dtype=torch.float32, device="cuda")
    input_copy = input_data.clone()

    # Create intermediate FFT result tensor
    fft_result_shape = (batch_size, fft_size_y, stride_y)
    fft_result = torch.zeros(fft_result_shape, dtype=torch.complex64, device="cuda")

    # Create output tensor for the recovered signal
    output_data = torch.zeros(input_shape, dtype=torch.float32, device="cuda")

    # Forward FFT
    zipfft.padded_rfft2d.fft(input_data, fft_result, fft_size_y, fft_size_x)

    # Inverse FFT
    zipfft.padded_rfft2d.ifft(fft_result, output_data, fft_size_y, fft_size_x)

    # Check that the output matches the input within tolerance
    # Note: Need to scale by the FFT size for proper normalization
    scale_factor = 1.0 / (fft_size_y * fft_size_x)
    output_data *= scale_factor

    assert torch.allclose(
        input_copy, output_data, atol=1e-3
    ), f"Round-trip padded FFT failed to recover original signal. Max diff: {torch.max(torch.abs(input_copy - output_data))}"
