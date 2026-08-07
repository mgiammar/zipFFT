# Installation

zipFFT compiles CUDA kernels at install time, so all installation paths require a working CUDA toolchain on the target machine.

Two installation paths are supported:

- **[Path 1 — conda recipe](#path-1--conda-recipe)**: conda manages the build-time dependencies (CUDA toolkit, cuFFTDx headers, PyTorch) and then compiles on your machine. Recommended for most users.
- **[Path 2 — pip from source](#path-2--pip-from-source)**: install PyTorch + `nvidia-mathdx` from PyPI, then `pip install`. Most flexible if you need to customize parts of the build process (e.g. custom CUDA flags, non-conda Python environment, etc.).

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

### 3. Install PyTorch and Python dependencies

cuFFTDx / MathDx headers are pulled in automatically via the `nvidia-mathdx` PyPI package
(a pure header-only wheel, no manual download needed). `setup.py` locates it at build time
through `nvidia.mathdx.__path__` and adds it to the include path.

```bash
pip install torch torchvision pytest pyyaml nvidia-mathdx
```

### 4. Install zipFFT

`setup.py` imports `torch` directly to configure the CUDA extension, so build with
`--no-build-isolation` to make sure it sees the PyTorch (and `nvidia-mathdx`) you just
installed into this environment rather than a fresh copy in an isolated build env:

```bash
pip install -e . --no-build-isolation
```

If you omit `--no-build-isolation`, pip will still auto-install `nvidia-mathdx` (declared in
`pyproject.toml`'s `[build-system] requires`) into its isolated build environment, but that
environment's PyTorch may not match the CUDA build you have installed.

To reduce compile time by targeting only your GPU's compute capability and skipping unused modules.

> Replace `8.9` with your GPU's SM version (e.g. `8.0` for A100, `9.0` for H100, `12.0` for Blackwell). If `CUDA_ARCHITECTURES` is not set, all supported architectures are compiled by default. Helpful list of compute architectures by GPU: [Arnon Shimoni - Matching CUDA arch and CUDA gencode for various NVIDIA architectures](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

```bash
CUDA_ARCHITECTURES=8.9 ENABLED_EXTENSIONS=padded_rconv2d pip install -e .
```

---
