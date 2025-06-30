# zipFFT
Efficient **z**ero **i**mplicitly **p**added (zip) FFT kernels in CUDA/cuFFTDx with a PyTorch interface.

## Rationale

The Discrete Fourier Transform (DFT), and more specifically the Fast Fourier Transform (FFT), is an invaluable algorithm in the space of signal processing.
One application is the FFT-based convolution which brings the computational complexity of a naive convolution from O(N^2) down to O(N logN).
The FFT also greatly benefits from massively parallel hardware like GPUs, and libraries like cuFFT expose general purpose FFT functionality which can run on GPU hardware.

Image processing frequently uses the FFT-based convolution (or cross-correlation) algorithm to compare regions of an image against some filter (kernel).
While general purpose FFT libraries parallelize these convolution operations, there are a handful of context where hardware utilization (and therefore efficiency) falls short.
The immediate example the zipFFT package seeks to fill is the case where a large image is being cross-correlated with a relatively small template.
The forward FFT of the small template requires zero-padding up to the same shape as the image, but this necessitates reading in and operating on a large portion of zero values.
Also, trips to and from global memory can be eliminated by including the point-wise multiplication operation in the FFT kernel itself.

By leveraging the cuFFTDx library, we can define and execute FFT operations within CUDA kernels.
We expose these kernels into Python-land through PyTorch linkage meaning these kernels can be executed on data held in `torch.Tensor` objects.
We also test the execution of each of these kernels against their PyTorch implementations for accuracy.



## Usage

The following code will execute a basic 1-dimensional complex-to-complex FFT using the zipFFT backend.
Note that the ordering of imports is important, and that the FFT operates in-place.

```python
import torch  # !!! NOTE !!! CUDA backend built by PyTorch needs torch imported first!
from zipfft import zipfft_binding

# Signal size must be in [16, 32, 64, 128, 256, 512, 1024]
x = torch.rand(512, dtype=torch.complex64, device="cuda")
zipfft_binding.fft_c2c_1d(x)

# Tensor 'x' now contains the FFT'd values
print(x)
```


## Installation

Currently requires compilation on host device.
It's recommended to use conda to manage which versions of CUDA-toolkit, MathDx, and other packages which are necessary.
Use the following steps to install the package from a fresh conda environment.

First, create a fresh conda environment and activate it
```bash
conda create -n zipfft python=3.12 -y && conda activate zipfft
```

Next, we use conda to install the required `MathDx` and `cuda-toolkit` library versions.
MathDx needs to be installed first, otherwise the conda solver will complain about platforms for `cuda-toolkit`; unsure why.
```bash
conda install conda-forge::mathdx
conda install nvidia/label/cuda-12.8.1::cuda-toolkit
```

PyTorch built with CUDA version 12.8 is required to compile the backend functions
```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 PyYAML
```

The package is then installable from source. Assuming you have cloned and navigated to the repo root directory, run the following to generate the pre-defined FFT implementation files and install the package under the current environment.
```bash
python generate_fft_configs.py
pip install -e .
```

## Further Information and Caveats

### Limitations

* Complex-to-complex FFT functions are executed in-place.
* Input size/shape needs to be known at compile time, so only common sets of FFT sizes are supported.
* Input data type also needs to be know at compile time, and no type casting is happening under-the-hood.

### Info on backend construction

The zipFFT backend is heavily templated C++/CUDA code which can be hard to parse at times.
These template constructions do permit reuse of kernels and launcher functions.
For example, the `zipfft.zipfft_binding.fft_c2c_1d` just calls one of the template instantiations at the bottom of [`src/cuda/src/cuda/complex_fft_1d.cu`](src/cuda/complex_fft_1d.cu).
<!-- Adding a new compiled FFT size would simply be a new line at the bottom of this file, for example a 2048-point FFT:
```c++
// ... existing code

template int block_complex_fft_1d<float2, 512u >(float2* data);
template int block_complex_fft_1d<float2, 1024u>(float2* data);
/* new */ template int block_complex_fft_1d<float2, 2048u >(float2* data);

// ... existing code
``` -->

Each kernel type (e.g. 1D FFT, 2D FFT, padded FFT kernels) are each contained within their own .cu file and exposed with a header into C++ land.
There is the script [`generate_fft_configs.py`](generate_fft_configs.py) which auto-generates these template implementations.

Many of the functional headers included in this library come directly from the CUDALibaraySamples repository or have been adapted therefrom.

## 🚧 Work in progress 🚧

These are a non-comprehensive, non-guaranteed list of future zipFFT functionality.

* float16/complex32 support
* float64/complex128 support
* Real-to-complex and complex-to-real kernels
* 1D padded FFT kernels
* 1D convolution (and padded convolution) kernels
* 2D FFT kernels
* 2D padded FFT kernels
* 2D convolution (and padded convolution) kernels
* Fused maximum cross-correlation kernels
