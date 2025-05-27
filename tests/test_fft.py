import torch  # !!! NOTE !! CUDA backend built by PyTorch needs torch imported first!
from zipfft import binding_cuda

def test_run_fft():
    try:
        binding_cuda.run_fft()
    except Exception as e:
        assert False, f"run_fft raised an exception: {e}"