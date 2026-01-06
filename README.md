# zipFFT
Efficient **z**ero **i**mplicitly **p**added (zip) FFT kernels in CUDA/cuFFTDx with a PyTorch interface.

What the zipFFT library is:

* A set of problem-specific CUDA kernels, leveraging cuFFTDx, for image convolution/cross-correlation.
* Header only, heavily templated implementations for integration with C++/CUDA.
* A means to accelerate two-dimensional template matching (2DTM) in cryo-EM for particular image shapes/sizes.
* Python bindings (built with PyTorch) for easy integration into PyTorch-based pipelines.

What the zipFFT library _is not_ (and does not intend to be):

* A general purpose FFT library.
* A replacement for established libraries like FFTW or cuFFT.
* A solution for all combinations of convolution/cross-correlation problem shapes/sizes.

## Rationale

The Discrete Fourier Transform (DFT), and more specifically the Fast Fourier Transform (FFT), is an invaluable algorithm in signal processing.
One application of the FFT, which leverages the [convolution theorem](https://en.wikipedia.org/wiki/Convolution_theorem), is to compute convolutions (or cross-correlations) in the frequency domain; the FFT-based convolution brings the complexity down from $O(N^2)$ to $O(N \log N)$ for 1-dimensional signals.

### The problem

Image processing frequently uses FFT-based convolutions (or cross-correlations) for pattern recognition, and general purpose FFT libraries -- like FFTW and cuFFT -- expose nice APIs for calling FFTs within image analysis pipelines.
One such pipeline is Two-Dimensional Template Matching (2DTM) in the field of cryo-EM where millions of these cross-correlations are computed to detect protein structures within noisy images of cells ([ref 1](https://elifesciences.org/articles/25648), [ref 2](https://www.biorxiv.org/content/10.1101/2025.08.26.672452v1)).
However, these general purpose libraries lack the granularity to exploit problem-specific structure to better utilize hardware and improve efficiency.
In 2DTM, the most obvious structures are 1)  large amount of zero-padding applied when calculating the FFT of a small projection and 2) fusing point-wise multiplications with FFT/IFFT kernels to mask unnecessary global memory reads/writes.

### The solution

The zipFFT library targets this specific problem structure to provide highly efficient kernels for executing these image convolutions/cross-correlations.
Data is selectively read from global memory, placing zero values into registers when reading from zero-padded regions of the input, through custom load operations, and point-wise multiplications are fused into the FFT/IFFT kernels to avoid unnecessary global memory traffic.

These kernels are also exposed into Python-land through PyTorch linkage for easy integration into existing PyTorch-based pipelines.
This also makes unit testing against the `pytorch.fft` module straightforward to ensure accuracy of all custom implementations.


## Installation

The `cuFFTDx` and associated `MathDx` libraries from NVIDIA are still under active development, so the following installation steps may be unstable.
We currently recommend using `conda` to manage dependencies between the packages and libraries, although this may be different on your system.

We are also working on providing easier installation methods for zipFFT through package managers in the future.

### 1. Create a new conda environment

```bash
conda create -n zipfft python=3.13 -y && conda activate zipfft
```

### 2. Install the CUDA toolkit package

zipFFT compiles CUDA code which gest linked to Python throught `pybind11` and PyTorch.
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

### 3. Install the MathDx/cuFFTDx libraries

Follow the instructions on the [cuFFTDx Download page](https://developer.nvidia.com/cufftdx-downloads) to download and install the `MathDx` and `cuFFTDx` libraries.
For example downloading the tarball for CUDA 12.x, extracting it, and moving the headers to `$CONDA_PREFIX/include/` would look like:

```bash
wget https://developer.nvidia.com/downloads/compute/cuFFTDx/redist/cuFFTDx/cuda13/nvidia-mathdx-25.06.1-cuda13.tar.gz
tar -xzf nvidia-mathdx-*.tar.gz
```

Next, move the include files to the conda environment's include directory

```bash
mv nvidia-mathdx-25.06.1/nvidia/mathdx/25.06/include/* $CONDA_PREFIX/include/
```

Optionally, remove the rest of the extracted files and tarball

```bash
rm -rf nvidia-mathdx-25.06.1 nvidia-mathdx-*.tar.gz
```

### 4. Install necessary PyTorch version and other Python dependencies

```bash
python -m pip install torch torchvision
python -m pip install pytest pyyaml
```

### 5. Run local install

Installation from source is currently the only way to install zipFFT.
However, all dependencies in the installation process should be automatically managed by zipFFT and conda.

```bash
pip install -e .
```

#### Targeting specific CUDA architectures and modules to speed up compilation

Compiling the entire zipFFT package can take a long time because of the heavily templated nature of the code and the additional testing/development specific modules compiled by default.
Selecting only a specific CUDA architecture and/or a subset of modules to compile can significantly speed up the installation process.
To select a CUDA architecture, set teh environment variable `CUDA_ARCHITECTURES` to a comma-separated list of architectures before running the installation.

```bash
export CUDA_ARCHITECTURES=8.9  # For Ada generation GPUs
```

Similarly, a subset of modules can be selected by setting the `ENABLED_EXTENSIONS` environment variable to a comma-separated list of module names.
If you are only interested in the 2D convolution/cross-correlation functions, you can set this variable as follows:

```bash
export ENABLED_EXTENSIONS=padded_rconv2d
```

## Running unit tests

Unit tests can be run through `pytest` after installation.

```bash
pytest
```

Any modules which were not compiled during installation (due to `ENABLED_EXTENSIONS` settings) will be skipped during testing.
