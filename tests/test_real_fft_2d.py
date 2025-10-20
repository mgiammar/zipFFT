"""Real-to-complex and complex-to-real 2D FFT tests for cuFFTDx comparing against PyTorch."""

import torch  # !!! NOTE !!! CUDA backend built by PyTorch needs torch imported first!
import zipfft

import pytest

# Skip entire module if rfft2d is not available
pytestmark = pytest.mark.skipif(
    not zipfft.is_extension_available("rfft2d"),
    reason="rfft2d extension not available",
)

TYPE_MAP = {
    "float32": torch.float32,
    "complex64": torch.complex64,
}

# Only get configs if rfft2d is available
if zipfft.rfft2d is not None:
    ALL_CONFIGS = zipfft.rfft2d.get_supported_fft_configs()
    FORWARD_FFT_CONFIGS = [
        (cfg[0], cfg[1], cfg[2], cfg[3]) for cfg in ALL_CONFIGS if cfg[4] is True
    ]
    INVERSE_FFT_CONFIGS = [
        (cfg[0], cfg[1], cfg[2], cfg[3]) for cfg in ALL_CONFIGS if cfg[4] is False
    ]
else:
    FORWARD_FFT_CONFIGS = []
    INVERSE_FFT_CONFIGS = []

# Currently only testing with float32/complex64
DATA_TYPES = [torch.float32]

# How many times to repeat each test to catch any non-deterministic issues
NUM_TEST_REPEATS = 10


def run_forward_real_fft_2d_test(
    n_cols: int,  # fft_size_x
    n_rows: int,  # fft_size_y
    stride_y: int,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32,
    atol: float = 1e-3,  # NOTE: Looser tolerance for accumulated multi-dimensional FFTs
):
    """Runs a single forward real 2D FFT test for given parameters.

    Parameters
    ----------
    n_cols : int
        Number of columns in the 2D array (size of FFT along first dimension).
    n_rows : int
        Number of rows in the 2D array (size of FFT along second dimension).
    stride_y : int
        The stride size (second dimension length).
    batch_size : int, optional
        The batch size, by default 1.
    dtype : torch.dtype, optional
        The data type of the input tensor, by default torch.float32.
    atol : float, optional
        Absolute tolerance for comparison, by default 1e-3.
    """
    # Create input tensor with appropriate shape
    input_shape = (n_rows, n_cols)
    output_shape = (n_rows, n_cols // 2 + 1)
    if batch_size > 1:
        input_shape = (batch_size, n_rows, n_cols)
        output_shape = (batch_size, n_rows, n_cols // 2 + 1)

    input_data = torch.randn(input_shape, dtype=dtype, device="cuda")
    input_data_clone = torch.empty_like(input_data).copy_(input_data)

    output_data = torch.zeros(output_shape, dtype=torch.complex64, device="cuda")

    # PyTorch reference: rfft2 along all dimensions
    torch_output = torch.fft.rfft2(input_data_clone)

    # Run our implementation
    zipfft.rfft2d.fft(input_data, output_data)

    # Verify results
    assert torch.allclose(
        torch_output, output_data, atol=atol
    ), f"Real 2D FFT results do not match ground truth. Max diff: {torch.max(torch.abs(torch_output - output_data))}"


def run_inverse_real_fft_2d_test(
    n_cols: int,
    n_rows: int,
    stride_y: int,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32,
    atol: float = 1e-3,  # NOTE: looser tolerance for accumulated multi-dimensional FFTs
):
    """Runs a single inverse real 2D FFT test for given parameters.

    Parameters
    ----------
    n_cols : int
        Number of columns in the 2D array (size of FFT along first dimension).
    n_rows : int
        Number of rows in the 2D array (size of FFT along second dimension).
    stride_y : int
        The stride size (second dimension length).
    batch_size : int, optional
        The batch size, by default 1.
    dtype : torch.dtype, optional
        The data type of the input tensor, by default torch.float32.
    atol : float, optional
        Absolute tolerance for comparison, by default 1e-3.
    """
    # Initial input data here is complex data with shape (n_rows, n_cols // 2 + 1)
    input_shape = (n_rows, n_cols // 2 + 1)
    output_shape = (n_rows, n_cols)
    if batch_size > 1:
        input_shape = (batch_size, n_rows, n_cols // 2 + 1)
        output_shape = (batch_size, n_rows, n_cols)

    # Start with random real data
    input_data = torch.randn(input_shape, dtype=torch.complex64, device="cuda")
    input_data_clone = torch.empty_like(input_data).copy_(input_data)

    # Create output tensor for inverse transform
    output_data = torch.zeros(output_shape, dtype=dtype, device="cuda")

    # PyTorch reference: irfft2
    torch_output = torch.fft.irfft2(input_data_clone, norm="forward")

    # Run our implementation
    zipfft.rfft2d.ifft(input_data, output_data)

    # Verify results
    assert torch.allclose(
        torch_output, output_data, atol=atol
    ), f"Inverse Real 2D FFT results do not match ground truth. Max diff: {torch.max(torch.abs(torch_output - output_data))}"


@pytest.mark.parametrize("n_rows,n_cols,stride_y,batch_size", FORWARD_FFT_CONFIGS)
@pytest.mark.parametrize("dtype", DATA_TYPES)
def test_real_fft_r2c_2d(n_rows, n_cols, stride_y, batch_size, dtype):
    """Test forward real 2D FFT for specific sizes, stride, batch size, and dtype."""
    for _ in range(NUM_TEST_REPEATS):
        run_forward_real_fft_2d_test(
            n_rows=n_rows,
            n_cols=n_cols,
            stride_y=stride_y,
            batch_size=batch_size,
            dtype=dtype,
        )


@pytest.mark.parametrize("n_rows,n_cols,stride_y,batch_size", INVERSE_FFT_CONFIGS)
@pytest.mark.parametrize("dtype", DATA_TYPES)
def test_real_fft_c2r_2d(n_rows, n_cols, stride_y, batch_size, dtype):
    """Test inverse real 2D FFT for specific sizes, stride, batch size, and dtype."""
    for _ in range(NUM_TEST_REPEATS):
        run_inverse_real_fft_2d_test(
            n_rows=n_rows,
            n_cols=n_cols,
            stride_y=stride_y,
            batch_size=batch_size,
            dtype=dtype,
        )
