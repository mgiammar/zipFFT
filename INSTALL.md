# Installation

zipFFT compiles CUDA kernels at install time, so all installation paths require a working CUDA toolchain on the target machine.

Two installation paths are supported:

- **[Path 1 — conda recipe](#path-1--conda-recipe)**: conda manages the build-time dependencies (CUDA toolkit, cuFFTDx headers, PyTorch) and then compiles on your machine. Recommended for most users.
- **[Path 2 — pip from source](#path-2--pip-from-source)**: manual dependency setup followed by `pip install`. Most flexible if you need to customize parts of the build process (e.g. custom CUDA flags, non-conda Python environment, etc.) but requires more manual steps.

Both paths compile the CUDA kernels locally, so the resulting binary is always matched to your GPU architecture and installed toolchain.

---

## Prerequisites (both paths)

- NVIDIA GPU with a supported compute capability (SM 8.0 – 12.0)
- CUDA Toolkit 12.x with `nvcc` on `PATH`
- Python >= 3.12

### Clone the repository

```bash
git clone https://github.com/mgiammar/zipFFT.git
cd zipFFT
```

### Create a new Python environment (optional but recommended)

Adjust this command depending on where you want to install zipFFT and what environment manager is available.

```bash
conda create -n zipfft python=3.14 -y
conda activate zipfft
```

### Select GPU architecture to build against (optional)

To speed up compilation and reduce binary size, set the `CUDA_ARCHITECTURES` environment variable to target only your GPU's compute capability. A list of compute architectures by GPU can be found here: [Arnon Shimoni - Matching CUDA arch and CUDA gencode for various NVIDIA architectures](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

```bash
export CUDA_ARCHITECTURES=8.9  # for Ada Lovelace GPUs (e.g. RTX A6000 ada)
# export CUDA_ARCHITECTURES=12.0  # for Blackwell GPUs (e.g. RTX 6000 blackwell)
```

---

## Path 1 — conda recipe

conda resolves cuFFTDx headers, PyTorch, etc. automatically before compiling zipFFT on your machine.

### 1. Install conda-build

```bash
conda install -y conda-build
```

### 2. Build the conda package

`conda build` must write to the base conda-bld directory so that `--use-local` can find the package regardless of which env is currently active. Pass `--croot` to enforce this:

```bash
conda build --croot $(conda info --base)/conda-bld conda-recipe/ -c nvidia -c pytorch -c conda-forge
```

### 3. Install the locally-built package with its dependencies

```bash
conda install --use-local zipfft -c pytorch -c nvidia -c conda-forge
```

## Verifying the installation

```bash
python -c "import zipfft; print('zipfft OK')"
pytest
```

Tests for any extension not compiled during installation are skipped automatically.

---

## Path 2 — pip from source

### 1. Create and activate an environment

```bash
conda create -n zipfft python=3.14 -y
conda activate zipfft
```

### 2. Install CUDA Toolkit

Confirm `nvcc` is available:

```bash
nvcc --version
```

If not, install the CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-toolkit) or ask your system administrator.

### 3. Install cuFFTDx / MathDx headers

Download the MathDx tarball for your CUDA version from the [cuFFTDx download page](https://developer.nvidia.com/cufftdx-downloads) and copy the headers into your environment:

```bash
wget https://developer.nvidia.com/downloads/compute/cuFFTDx/redist/cuFFTDx/cuda13/nvidia-mathdx-25.06.1-cuda13.tar.gz
tar -xzf nvidia-mathdx-25.06.1-cuda13.tar.gz
mv nvidia-mathdx-25.06.1/nvidia/mathdx/25.06/include/* $CONDA_PREFIX/include/
rm -rf nvidia-mathdx-25.06.1 nvidia-mathdx-*.tar.gz
```

### 4. Install PyTorch and Python dependencies

```bash
pip install torch torchvision pytest pyyaml
```

### 5. Install zipFFT

```bash
pip install -e .
```

To reduce compile time by targeting only your GPU's compute capability and skipping unused modules.

> Replace `8.9` with your GPU's SM version (e.g. `8.0` for A100, `9.0` for H100, `12.0` for Blackwell). If `CUDA_ARCHITECTURES` is not set, all supported architectures are compiled by default. Helpful list of compute architectures by GPU: [Arnon Shimoni - Matching CUDA arch and CUDA gencode for various NVIDIA architectures](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

```bash
CUDA_ARCHITECTURES=8.9 ENABLED_EXTENSIONS=padded_rconv2d pip install -e .
```

---
