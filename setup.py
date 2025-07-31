"""Setup script for binding C++/CUDA code with Python using pybind11."""

from setuptools import setup, Extension
import pybind11

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.0.2alpha"


# fmt: off
DEBUG_PRINT = False
if DEBUG_PRINT:
    print("Using pybind11 include directory: ", pybind11.get_include())
    print("Using torch include directory:    ", pybind11.get_include(user=True))
    print("Using torch library directory:    ", pybind11.get_cmake_dir())
    print("Using library dirs:               ", torch.utils.cpp_extension.CUDA_HOME)
    print("                                  ", torch.utils.cpp_extension.TORCH_LIB_PATH)
# fmt: on


DEFAULT_COMPILE_ARGS = {
    # "cxx": ["-O3"],
    "nvcc": [
        "-O3",
        "-std=c++17",
        # NOTE: Necessary to un-define PyTorch default macros with fp16/bf16 to get
        # cuFFTDx library to compile correctly.
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ],
}


complex_fft_1d_extension = CUDAExtension(
    name="zipfft.cfft1d",  # Module name needs to match source code PYBIND11 statement
    sources=["src/cuda/complex_fft_1d_binding.cu"],
    include_dirs=[pybind11.get_include()],
    extra_compile_args=DEFAULT_COMPILE_ARGS,
)


real_fft_1d_extension = CUDAExtension(
    name="zipfft.rfft1d",  # Module name needs to match source code PYBIND11 statement
    sources=["src/cuda/real_fft_1d_binding.cu"],
    include_dirs=[pybind11.get_include()],
    extra_compile_args=DEFAULT_COMPILE_ARGS,
)


padded_real_fft_1d_extension = CUDAExtension(
    name="zipfft.padded_rfft1d",  # Module name needs to match source code
    sources=["src/cuda/padded_real_fft_1d_binding.cu"],
    include_dirs=[pybind11.get_include()],
    extra_compile_args=DEFAULT_COMPILE_ARGS,
)

# TODO: Make this setup script more robust (plus conda recipe)
setup(
    ext_modules=[
        complex_fft_1d_extension,
        real_fft_1d_extension,
        padded_real_fft_1d_extension,
    ],
    cmdclass={"build_ext": BuildExtension},
    version=__version__,
)
