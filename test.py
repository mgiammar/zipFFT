import torch
from zipfft import fft_strided

FORWARD_FFT_CONFIGS = fft_strided.get_supported_sizes()

fft_size = 256

fft_shape = (1, fft_size, 2)

x0 = torch.randn(fft_shape, dtype=torch.complex64, device="cuda")
x1 = x0.clone()

x0 = torch.fft.fft(x0, dim=1)

# NOTE: This zipFFT function is in-place
fft_strided.fft(x1, True)

# convert torch tensors to numpy arrays

x0_numpy = x0.cpu().numpy()
x1_numpy = x1.cpu().numpy()


assert torch.allclose(x0, x1, atol=1e-4), "FFT results do not match ground truth"
