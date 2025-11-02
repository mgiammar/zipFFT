"""Setup script for binding C++/CUDA code with Python using pybind11."""

from setuptools import setup, Extension
import pybind11
import argparse
import sys
import os

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.0.2alpha"


# Parse command line arguments for CUDA architectures
def parse_cuda_architectures():
    """Parse CUDA architectures from command line arguments or environment variables."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--cuda-arch",
        "--cuda-architectures",
        dest="cuda_architectures",
        help='Comma-separated list of CUDA architectures to compile for (e.g., "7.5,8.0,8.6")',
        default=None,  # Will use env var or fallback if None
    )
    parser.add_argument(
        "--enable-extensions",
        dest="enable_extensions",
        help='Comma-separated list of extensions to build (e.g., "cfft1d,rfft1d,padded_rfft1d,padded_cfft1d,strided_cfft1d,rfft2d,padded_rfft2d,padded_rconv2d")',
        default=None,  # Will use env var or fallback if None
    )

    # Parse known args to avoid conflicts with setuptools
    args, unknown = parser.parse_known_args()

    # Remove our custom args from sys.argv so setuptools doesn't see them
    for arg_name in ["--cuda-arch", "--cuda-architectures", "--enable-extensions"]:
        if arg_name in sys.argv:
            idx = sys.argv.index(arg_name)
            sys.argv.pop(idx)  # Remove the argument
            # Remove value for non-flag arguments
            if idx < len(sys.argv):
                sys.argv.pop(idx)

    return args


# Parse arguments
parsed_args = parse_cuda_architectures()

# Get CUDA architectures with precedence: CLI args > env vars > defaults
cuda_archs_str = (
    parsed_args.cuda_architectures
    or os.environ.get("CUDA_ARCHITECTURES")
    or "8.0,8.6,8.9,9.0,12.0"
)
cuda_architectures = [arch.strip() for arch in cuda_archs_str.split(",")]

# Get enabled extensions with precedence: CLI args > env vars > defaults
enabled_exts_str = (
    parsed_args.enable_extensions
    or os.environ.get("ENABLED_EXTENSIONS")
    or "cfft1d,rfft1d,padded_rfft1d,padded_cfft1d,strided_cfft1d,rfft2d,padded_rfft2d,padded_rconv2d"
)
enabled_extensions = [ext.strip() for ext in enabled_exts_str.split(",")]


# fmt: off
DEBUG_PRINT = False
if DEBUG_PRINT:
    print("Using pybind11 include directory: ", pybind11.get_include())
    print("Using torch include directory:    ", pybind11.get_include(user=True))
    print("Using torch library directory:    ", pybind11.get_cmake_dir())
    print("Using library dirs:               ", torch.utils.cpp_extension.CUDA_HOME)
    print("                                  ", torch.utils.cpp_extension.TORCH_LIB_PATH)
    print("CUDA architectures to compile:    ", cuda_architectures)
# fmt: on


def get_compile_args():
    """Generate compile arguments including CUDA architectures."""
    nvcc_args = [
        "-O3",
        "-std=c++17",
        # NOTE: Necessary to un-define PyTorch default macros with fp16/bf16 to get
        # cuFFTDx library to compile correctly.
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ]

    # Add preprocessor definitions for enabled CUDA architectures
    arch_defines = []
    for arch in cuda_architectures:
        # Convert decimal arch to what cuFFTDx expects
        # e.g. 8.9 -> 89 -> 890
        # e.g. 12.0 -> 120 -> 1200
        arch_int = arch.replace(".", "")
        arch_int = arch_int + "0"

        arch_defines.append(f"-DENABLE_CUDA_ARCH_{arch_int}")

    nvcc_args.extend(arch_defines)

    # Add architecture-specific flags
    for arch in cuda_architectures:
        nvcc_args.extend(
            [
                "-gencode",
                f"arch=compute_{arch.replace('.', '')},code=sm_{arch.replace('.', '')}",
            ]
        )

    return {
        # "cxx": ["-O3"],
        "nvcc": nvcc_args,
    }


DEFAULT_COMPILE_ARGS = get_compile_args()

# Conditionally create extensions
ext_modules = []

if "cfft1d" in enabled_extensions:
    complex_fft_1d_extension = CUDAExtension(
        name="zipfft.cfft1d",
        sources=["src/cuda/complex_fft_1d_binding.cu"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=DEFAULT_COMPILE_ARGS,
    )
    ext_modules.append(complex_fft_1d_extension)

if "rfft1d" in enabled_extensions:
    real_fft_1d_extension = CUDAExtension(
        name="zipfft.rfft1d",
        sources=["src/cuda/real_fft_1d_binding.cu"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=DEFAULT_COMPILE_ARGS,
    )
    ext_modules.append(real_fft_1d_extension)

if "padded_rfft1d" in enabled_extensions:
    padded_real_fft_1d_extension = CUDAExtension(
        name="zipfft.padded_rfft1d",
        sources=["src/cuda/padded_real_fft_1d_binding.cu"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=DEFAULT_COMPILE_ARGS,
    )
    ext_modules.append(padded_real_fft_1d_extension)

if "padded_cfft1d" in enabled_extensions:
    padded_complex_fft_1d_extension = CUDAExtension(
        name="zipfft.padded_cfft1d",
        sources=["src/cuda/padded_complex_fft_1d_binding.cu"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=DEFAULT_COMPILE_ARGS,
    )
    ext_modules.append(padded_complex_fft_1d_extension)

if "strided_cfft1d" in enabled_extensions:
    strided_complex_fft_1d_extension = CUDAExtension(
        name="zipfft.strided_cfft1d",
        sources=["src/cuda/strided_complex_fft_1d_binding.cu"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=DEFAULT_COMPILE_ARGS,
    )
    ext_modules.append(strided_complex_fft_1d_extension)

# NOTE: Removed in favor of testing against 2D padded FFTs
# if "strided_padded_cfft1d" in enabled_extensions:
#     strided_padded_complex_fft_1d_extension = CUDAExtension(
#         name="zipfft.strided_padded_cfft1d",
#         sources=["src/cuda/strided_padded_complex_fft_1d_binding.cu"],
#         include_dirs=[pybind11.get_include()],
#         extra_compile_args=DEFAULT_COMPILE_ARGS,
#     )
#     ext_modules.append(strided_padded_complex_fft_1d_extension)

if "rfft2d" in enabled_extensions:
    real_fft_2d_extension = CUDAExtension(
        name="zipfft.rfft2d",
        sources=["src/cuda/real_fft_2d_binding.cu"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=DEFAULT_COMPILE_ARGS,
    )
    ext_modules.append(real_fft_2d_extension)

if "padded_rfft2d" in enabled_extensions:
    padded_real_fft_2d_extension = CUDAExtension(
        name="zipfft.padded_rfft2d",
        sources=["src/cuda/padded_real_fft_2d_binding.cu"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=DEFAULT_COMPILE_ARGS,
    )
    ext_modules.append(padded_real_fft_2d_extension)

if "padded_rconv2d" in enabled_extensions:
    padded_real_conv_2d_extension = CUDAExtension(
        name="zipfft.padded_rconv2d",
        sources=["src/cuda/real_conv_2d_binding.cu"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=DEFAULT_COMPILE_ARGS,
    )
    ext_modules.append(padded_real_conv_2d_extension)

# Write build configuration to a file for testing
build_config = {
    "cuda_architectures": cuda_architectures,
    "enabled_extensions": enabled_extensions,
}

os.makedirs("src/zipfft", exist_ok=True)
with open("src/zipfft/build_config.py", "w") as f:
    f.write(f"# Auto-generated build configuration\n")
    f.write(f"CUDA_ARCHITECTURES = {cuda_architectures}\n")
    f.write(f"ENABLED_EXTENSIONS = {enabled_extensions}\n")

# TODO: Make this setup script more robust (plus conda recipe)
setup(
    name="zipFFT",
    description="Custom FFT operations for PyTorch using cuFFTDx",
    author="Matthew Giammar",
    python_requires=">=3.9",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    version=__version__,
)
