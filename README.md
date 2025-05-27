# zipFFT
Efficient zero implicit padded (zip) FFT kernels in CUDA with a PyTorch interface.

## Installation

Currently requires compilation on host device (which is assumed to be a Linux device).
It's recommended to use conda to manage which versions of CUDA-toolkit, MathDx, and other packages which are necessary.
Use the following steps to install the package from a fresh conda environment.

First, create a fresh conda environment and activate it
```
conda create -n zipfft python=3.12 -y && conda activate zipfft
```

Next, we use conda to install the required `MathDx` and `cuda-toolkit` library versions.
MathDx needs to be installed first, otherwise the conda solver will complain about platforms for `cuda-toolkit`; unsure why.
```
conda install conda-forge::mathdx
conda install nvidia/label/cuda-12.8.1::cuda-toolkit
```

PyTorch built with CUDA version 12.8 is required to compile the backend functions
```
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
