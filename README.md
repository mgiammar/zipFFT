# zipFFT

Efficient **z**ero **i**mplicitly **p**added (zip) FFT kernels written in CUDA/cuFFTDx with a PyTorch interface.

What the zipFFT library is:

* A set of problem-specific CUDA kernels, leveraging cuFFTDx, for image convolution/cross-correlation.
* Header only, heavily templated implementations for integration with C++/CUDA.
* Python bindings (built with PyTorch) for easy integration into PyTorch-based pipelines.

What the zipFFT library _is not_ (and does not intend to be):

* A general purpose FFT library.
* A replacement for libraries like [FFTW](https://www.fftw.org) or [cuFFT](https://developer.nvidia.com/cufft).
* A solution for all combinations of convolution/cross-correlation problem shapes/sizes.

## Rationale for zipFFT

The Discrete Fourier Transform (DFT), and more specifically the Fast Fourier Transform (FFT), is an invaluable algorithm in signal processing.
One application leverages the [convolution theorem](https://en.wikipedia.org/wiki/Convolution_theorem) by computing convolutions (or cross-correlations) in the frequency domain; the FFT-based convolution brings the complexity down from $O(N^2)$ to $O(N \log N)$ for 1-dimensional signals.

Popular FFT libraries (like FFTW and cuFFT) are well-established in image analysis pipelines, and FFT computation can be accelerated on highly parallel GPUs.
However, these general purpose libraries lack the granularity to maximize hardware and software performance for domain-specific problems.
One such problem, which motivated the development of zipFFT, is Two-Dimensional Template Matching (2DTM) in the field of cryo-EM where millions of FFT-based cross-correlations are computed to detect protein structures within nanometer-scale images of cells ([ref 1](https://doi.org/10.1107/S2059798325009982)).

### Zero-padding signals for large image cross-correlation

In 2DTM, small macromolecule projections (from 256×256 to 512×512 pixels) are zero-padded to match a much larger image (typically 4096×4096 pixels) before computing a 2D FFT.
Standard GPU libraries like PyTorch's `torch.fft` explicitly allocate and copy zero-padded arrays before calling the FFT routine.
This is inefficient because:

1. **Predictable structure**: Image and projection shapes are fixed, so padded regions are known at compile-time
2. **Wasted bandwidth**: Only 0.39–1.6% of global memory reads contain non-zero data
3. **Missed optimization opportunities**: General-purpose libraries can't exploit this problem structure

zipFFT exploits this problem structure to make a more efficient cross-correlation pipeline by zero-padding _implicitly_ during the load step:

```c++
if (index < SignalLength)
    register_data = global_memory[index];  // Read actual data
else
    register_data = 0.0;                   // Skip memory read entirely
```

### Fused convolution kernel

A second optimization which zipFFT makes is fusing the frequency-domain multiplication kernel with the forward/inverse FFT kernels.
Normally a convolution pipeline would do the following steps:

1. Read in image data from global memory
2. Compute the forward FFT on said data (`img_fft = FFT(img)`)
3. Write FFT results back to global memory
4. Read in frequency-domain image + kernel data
5. Compute point-wise multiplication (`corr_fft = img_fft * kernel`)
6. Write results to global memory
7. Read results back into registers
8. Compute the inverse FFT (`corr = IFFT(corr_fft)`)
9. Write results back into global memory

Steps 3+4 and 6+7 do redundant read/writes to/from global memory; the data from the previous computation step is immediately used again.
By designing a single kernel that does `FFT + multiplication + IFFT`, zipFFT further saves global memory transactions.

<!-- 
### The problem

Image processing frequently uses FFT-based convolutions (or cross-correlations) for pattern recognition, and general purpose FFT libraries -- like FFTW and cuFFT -- expose nice APIs for calling FFTs within image analysis pipelines.
One such pipeline is Two-Dimensional Template Matching (2DTM) in the field of cryo-EM where millions of these cross-correlations are computed to detect protein structures within noisy images of cells ([ref 1](https://elifesciences.org/articles/25648), [ref 2](https://www.biorxiv.org/content/10.1101/2025.08.26.672452v1)).
However, these general purpose libraries lack the granularity to exploit problem-specific structure to better utilize hardware and improve efficiency.
In 2DTM, the most obvious structures are 1)  large amount of zero-padding applied when calculating the FFT of a small projection and 2) fusing point-wise multiplications with FFT/IFFT kernels to mask unnecessary global memory reads/writes.

### The solution

The zipFFT library targets this specific problem structure to provide highly efficient kernels for executing these image convolutions/cross-correlations.
Data is selectively read from global memory, placing zero values into registers when reading from zero-padded regions of the input, through custom load operations, and point-wise multiplications are fused into the FFT/IFFT kernels to avoid unnecessary global memory traffic.

These kernels are also exposed into Python-land through PyTorch linkage for easy integration into existing PyTorch-based pipelines.
This also makes unit testing against the `pytorch.fft` module straightforward to ensure accuracy of all custom implementations. -->

## Installation

> **Note**: cuFFTDx is under active development. These instructions were tested with CUDA 12.x and MathDx 25.06.
> **NOTE**: zipFFT is also under development and not widely tested across systems. Your mileage may vary with the following instructions.

### Prerequisites

* CUDA Toolkit installed and `nvcc` available on PATH
* conda (recommended) or another dependency and environment manager

### Quick Start

<!-- The `cuFFTDx` and associated `MathDx` libraries from NVIDIA are still under active development, so the following installation steps may be unstable.
We currently recommend using `conda` to manage dependencies between the packages and libraries, although this may be different on your system. -->

<!-- We are also working on providing easier installation methods for zipFFT through package managers in the future. -->

#### 1. Create a new conda environment

```bash
conda create -n zipfft python=3.13 -y && conda activate zipfft
```

#### 2. Install the CUDA toolkit package

zipFFT compiles CUDA code which gest linked to Python though `pybind11` and PyTorch.
This compilation step requires the CUDA toolkit to be installed on your system.
Please see [NVIDIA Cuda toolkit](https://developer.nvidia.com/cuda-toolkit) for information about installing the CUDA toolkit on your system.
Or contact your system administrator for help installing the CUDA toolkit.

The `nvcc` compiler needs discoverable as an executable on your system PATH for the installation to succeed.
Make sure the MathDx/cuFFTDx libraries (next step) match the CUDA toolkit version version.

```bash
# Find the nvcc compiler version
nvcc --version
```

<!-- We have tested zipFFT with CUDA 12.9, but newer versions may be found on the (anaconda)[https://anaconda.org/nvidia/cuda-toolkit] website.
A different version of CUDA may be installed on your system, and you should update the version accordingly

```bash
conda install nvidia/label/cuda-12.9.1::cuda-toolkit
# conda install nvidia/label/cuda-12.9.1::cuda-toolkit
``` -->

#### 3. Install the MathDx/cuFFTDx libraries

Follow the instructions on the [cuFFTDx Download page](https://developer.nvidia.com/cufftdx-downloads) to download and install the `MathDx` and `cuFFTDx` libraries.
For example downloading the tarball for CUDA 12.x, extracting it, and moving the headers to `$CONDA_PREFIX/include/` would look like:

```bash
# Download MathDx/cuFFTDx headers
wget https://developer.nvidia.com/downloads/compute/cuFFTDx/redist/cuFFTDx/cuda13/nvidia-mathdx-25.06.1-cuda13.tar.gz
tar -xzf nvidia-mathdx-*.tar.gz
```

Next, move the include files to the conda environment's include directory

```bash
mv nvidia-mathdx-25.06.1/nvidia/mathdx/25.06/include/* $CONDA_PREFIX/include/
```

(optional) remove the rest of the extracted files and tarball

```bash
rm -rf nvidia-mathdx-25.06.1 nvidia-mathdx-*.tar.gz
```

#### 4. Install necessary PyTorch version and other Python dependencies

```bash
python -m pip install torch torchvision
python -m pip install pytest pyyaml
```

#### 5. Run local install

Installation from source is currently the only way to install zipFFT.
However, all dependencies in the installation process should be automatically managed by zipFFT and conda.

```bash
pip install -e .
```

### Targeting specific CUDA architectures and modules to speed up compilation

The heavily-templated code can take a long time to compile.
To speed this up, specify your GPU architecture and/or limit which modules are built:

<!-- 
Compiling the entire zipFFT package can take a long time because of the heavily templated nature of the code and the additional testing/development specific modules compiled by default.
Selecting only a specific CUDA architecture and/or a subset of modules to compile can significantly speed up the installation process.
To select a CUDA architecture, set teh environment variable `CUDA_ARCHITECTURES` to a comma-separated list of architectures before running the installation. -->

```bash
# Target only a specific GPU architecture
export CUDA_ARCHITECTURES=8.9  # For Ada generation GPUs
```

```bash
# Build only the 2D convolution module
export ENABLED_EXTENSIONS=padded_rconv2d
```

Then run the installation again

```bash
pip install -e .
```

<!-- Similarly, a subset of modules can be selected by setting the `ENABLED_EXTENSIONS` environment variable to a comma-separated list of module names.
If you are only interested in the 2D convolution/cross-correlation functions, you can set this variable as follows:

```bash
export ENABLED_EXTENSIONS=padded_rconv2d
``` -->

## Brief usage example

zipFFT only supports specific pre-compiled shape configurations.
You can query available configurations:

```python
import torch
import zipfft

# List all supported (signal_y, signal_x, fft_y, fft_x, batch, is_cross_corr) configs
configs = zipfft.padded_rconv2d.get_supported_conv_configs()
print(configs[:5])  # Show first 5 configurations
```

### Cross-correlation example

```python
import torch
import zipfft

# Configuration must match a compiled shape
signal_y, signal_x = 512, 512
fft_y, fft_x = 4096, 4096
batch_size = 4
device = "cuda"

# Input image (pre-transformed to frequency domain)
image = torch.randn(fft_y, fft_x, device=device)
image_fft = torch.fft.rfftn(image)

# NOTE: zipFFT expects transposed layout: (fft_x // 2 + 1, fft_y)
image_fft = image_fft.transpose(-2, -1).contiguous()

# Filter/template to cross-correlate
kernel = torch.randn(signal_y, signal_x, device=device)

# Allocate workspace and output
workspace = torch.empty(fft_y, fft_x // 2 + 1, dtype=torch.complex64, device=device)
output = torch.empty(fft_y - signal_y + 1, fft_x - signal_x + 1, device=device)

# Run cross-correlation
zipfft.padded_rconv2d.corr(kernel, workspace, image_fft, output, fft_y, fft_x)
```

### Equivalent PyTorch reference

```python
# Same computation using PyTorch (for comparison/validation)
kernel_fft = torch.fft.rfftn(kernel, s=(fft_y, fft_x))
result_fft = image_fft.transpose(-2, -1) * torch.conj(kernel_fft)
result = torch.fft.irfftn(result_fft)
result = result[:fft_y - signal_y + 1, :fft_x - signal_x + 1]
```

> **Note**: Use `zipfft.padded_rconv2d.conv()` for convolution (without conjugation).

## Running unit tests

Unit tests can be run through `pytest` after installation.

```bash
pytest
```

Any modules which were not compiled during installation (due to `ENABLED_EXTENSIONS` settings) will be skipped during testing.

## Adding support for more shapes/sizes

zipFFT compiles specific shape configurations at build time. The supported configurations are defined in:

```
src/cuda/real_conv_2d_binding.cu
```

To add a new configuration, edit the `SUPPORTED_CONV_CONFIGS` array:

```cpp
static constexpr std::array<...> SUPPORTED_CONV_CONFIGS = {{
    // Format: (signal_y, signal_x, fft_y, fft_x, batch, cross_correlate)
    {512, 512, 4096, 4096, 1, true},   // Example: 512×512 template in 4096×4096 image
    {256, 256, 2048, 2048, 8, true},   // Add your configuration here
    // ...
}};
```

Then rebuild:

```bash
pip install -e .
```

### cuFFTDx size limitations

Note that **cuFFTDx does not support arbitrary FFT sizes**. Supported sizes depend on your GPU architecture and are generally:

Refer to the [cuFFTDx documentation](https://docs.nvidia.com/cuda/cufftdx/index.html) for size support on your target architecture.
