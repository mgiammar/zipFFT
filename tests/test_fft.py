import torch  # !!! NOTE !!! CUDA backend built by PyTorch needs torch imported first!
from zipfft import binding_cuda

def test_fft_c2c_1d():
    """Run basic tests associated with the 1D C2C FFT."""
    x = torch.randn(128, dtype=torch.complex64, device="cuda")
    
    # Ground truth to compare against
    y_gt = torch.fft.fft(x, norm="forward")

    # Run the custom FFT implementation (in-place operation)
    binding_cuda.fft_c2c_1d(x)
    y = x.clone()
    
    ### DEBUG: Print the difference ###
    print("Difference between custom FFT and ground truth:", torch.abs(y - y_gt).max().item())

    assert y.shape == y_gt.shape, "Output shape mismatch"
    assert torch.allclose(y, y_gt, atol=1e-6), "FFT results do not match ground truth"