# Documentation for the zipFFT package

zipFFT seeks to accelerate a subset of FFT-based convolution algorithms in the cryo-EM image processing field using the NVIDIA cuFFTDx library.
zipFFT therefore is _not_ a general purpose FFT library, but it could be extended to include more functionality
This brief documentation describes the usage of zipFFT, some of the underlying theory, and implementation details which may be of interest to developers looking to use or extend zipFFT.

- [Memory and I/O Patterns](memory_io_patterns.md)

## Using zipFFT with PyTorch

zipFFT includes Python bindings to execute algorithms from PyTorch CUDA tensors.
This allows PyTorch to manage the complexities of memory allocation and signal pre-processing while maintaining the high-performance backend.
Currently, the library includes the following execution methods, most of which are for verifying the accuracy of algorithm implementations (correctness is critical for the library).

1. **Complex-to-Complex FFT (1D)**

    In-place transformation of a complex64 input tensor using a 1D C2C FFT.

   - Forward: `zipfft.cfft1d.fft(input_tensor)` - In-place operation
   - Inverse: `zipfft.cfft1d.ifft(input_tensor)` - In-place operation

2. **Real-to-Complex FFT (1D)**

    Out-of-place transformation of a float32 `input_tensor` to a complex64 `output_tensor` using a 1D R2C FFT.
    Or the inverse transformation from a complex64 `input_tensor` to a float32 `output_tensor` using a 1D C2R FFT.

   - Forward: `zipfft.rfft1d.rfft(input_tensor, output_tensor)`
   - Inverse: `zipfft.rfft1d.irfft(input_tensor, output_tensor)`

3. **Padded Real-to-Complex FFT (1D)**

    Out-of-place transformation of a float32 `input_tensor` to a complex64 `output_tensor` using a 1D R2C FFT with zero-padding to `fft_size`.
    Or the inverse transformation from a complex64 `input_tensor` to a float32 `output_tensor` using a 1D C2R FFT where the output signal is _cropped_ (not zero-padded) to `fft_size`.

   - Forward: `zipfft.padded_rfft1d.prfft(input_tensor, output_tensor, fft_size)`
   - Inverse: `zipfft.padded_rfft1d.pirfft(input_tensor, output_tensor, fft_size)`

4. **Padded Complex-to-Complex FFT (1D)**

    Same as above, but for complex-to-complex transforms.

    In-place transformation of a complex64 `input_tensor` using a 1D C2C FFT with zero-padding to `fft_size`.
    Or the inverse transformation from a complex64 `input_tensor` using a 1D C2C FFT with zero-padding to `fft_size`.

   - Forward: `zipfft.padded_cfft1d.pcfft(input_tensor, fft_size)` - In-place operation
   - Inverse: `zipfft.padded_cfft1d.picfft(input_tensor, fft_size)` - In-place operation

5. **Strided Complex FFT (1D)**

    Transforms which operate not on the "fastest" dimension of the input tensor.
    Can be used to build up higher-dimensional FFTs.

6. **Strided Padded Complex FFT (1D)**
    
    Same as above, but with zero-padding.

### Example: complex-to-complex FFT

```python
import torch  # NOTE: must import torch before zipfft
import zipfft

# Create a random 1D complex tensor of size 1024
x = torch.randn(1024, dtype=torch.complex64, device="cuda")

# Perform an in-place forward FFT
zipfft.cfft1d.fft(x)  # x now contains the FFT result
```

### Note on supported sizes/shapes/paddings

zipFFT is a heavily templated library, partially because cuFFTDx itself requires compile-time knowledge of the FFT size and other configurations.
A limited set of shapes/sizes, defined at compile time, are supported.
 
🚧 We are working to allow users to define the desired sizes/shapes/paddings when installing zipFFT. 🚧

## Using zipFFT with C++ / CUDA

Backend functions are exposed via header `.cuh` files in the `src/cuda/` directory and have particular template definitions.
These templates generally include the pointer types, FFT size, if the FFT is forward or inverse, and some cuFFTDx-specific configuration parameters.
These functions can be directly included in CUDA code without the Python bindings.

🚧 We are working on building out a more complete API documentation 🚧