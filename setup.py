"""Setup script for binding C++/CUDA code with Python using pybind11."""

from setuptools import setup, Extension
import pybind11
import os

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.0.0"

abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)

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
    "cxx": ["-O3", "-std=c++17", f"-D_GLIBCXX_USE_CXX11_ABI={abi}"],
    "nvcc": [
        "-O3",
        "-std=c++17",
        f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ],
}

torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")

full_include_dirs = [
    pybind11.get_include(),
    "/home/shaharsandhaus/nvidia-mathdx-25.06.1/nvidia/mathdx/25.06/include",
    "/home/shaharsandhaus/cutlass/include"
]

fft_nonstrided_extension = CUDAExtension(
    name="zipfft.fft_nonstrided",  # Module name needs to match source code PYBIND11 statement
    sources=["src/cuda/fft_nonstrided.cu"],
    include_dirs=full_include_dirs,
    extra_compile_args=DEFAULT_COMPILE_ARGS,
    runtime_library_dirs=[torch_lib],
    extra_link_args=[f"-Wl,-rpath,{torch_lib}"],         # belt & suspenders
    libraries=["c10", "torch_cpu", "torch_python"],      # pull in the symbols
)

fft_nonstrided_padded_extension = CUDAExtension(
    name="zipfft.fft_nonstrided_padded",  # Module name needs to match source code
    sources=["src/cuda/fft_nonstrided_padded.cu"],
    include_dirs=full_include_dirs,
    extra_compile_args=DEFAULT_COMPILE_ARGS,
    runtime_library_dirs=[torch_lib],
    extra_link_args=[f"-Wl,-rpath,{torch_lib}"],         # belt & suspenders
    libraries=["c10", "torch_cpu", "torch_python"],      # pull in the symbols
)

fft_strided_extension = CUDAExtension(
    name="zipfft.fft_strided",  # Module name needs to match source code PYBIND11 statement
    sources=["src/cuda/fft_strided.cu"],
    include_dirs=full_include_dirs,
    extra_compile_args=DEFAULT_COMPILE_ARGS,
    runtime_library_dirs=[torch_lib],
    extra_link_args=[f"-Wl,-rpath,{torch_lib}"],         # belt & suspenders
    libraries=["c10", "torch_cpu", "torch_python"],      # pull in the symbols
)

complex_conv_1d_strided_extension = CUDAExtension(
    name="zipfft.conv1d_strided",  # Module name needs to match source code PYBIND11 statement
    sources=["src/cuda/convolution_strided_binding.cu"],
    include_dirs=full_include_dirs,
    extra_compile_args=DEFAULT_COMPILE_ARGS,
    runtime_library_dirs=[torch_lib],
    extra_link_args=[f"-Wl,-rpath,{torch_lib}"],         # belt & suspenders
    libraries=["c10", "torch_cpu", "torch_python"],      # pull in the symbols
)

complex_conv_1d_strided_padded_extension = CUDAExtension(
    name="zipfft.conv1d_strided_padded",  # Module name needs to match source code PYBIND11 statement
    sources=["src/cuda/convolution_strided_padded_binding.cu"],
    include_dirs=full_include_dirs,
    extra_compile_args=DEFAULT_COMPILE_ARGS,
    runtime_library_dirs=[torch_lib],
    extra_link_args=[f"-Wl,-rpath,{torch_lib}"],         # belt & suspenders
    libraries=["c10", "torch_cpu", "torch_python"],      # pull in the symbols
)

# TODO: Make this setup script more robust (plus conda recipe)
setup(
    ext_modules=[
        fft_nonstrided_extension,
        fft_nonstrided_padded_extension,
        fft_strided_extension,
        complex_conv_1d_strided_extension,
        complex_conv_1d_strided_padded_extension
    ],
    cmdclass={"build_ext": BuildExtension},
    version=__version__,
)
