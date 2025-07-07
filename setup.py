"""Setup script for binding C++/CUDA code with Python using pybind11."""

from setuptools import setup, Extension
import pybind11

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.0.1alpha"


# fmt: off
DEBUG_PRINT = False
if DEBUG_PRINT:
    print("Using pybind11 include directory: ", pybind11.get_include())
    print("Using torch include directory:    ", pybind11.get_include(user=True))
    print("Using torch library directory:    ", pybind11.get_cmake_dir())
    print("Using library dirs:               ", torch.utils.cpp_extension.CUDA_HOME)
    print("                                  ", torch.utils.cpp_extension.TORCH_LIB_PATH)
# fmt: on

# TODO: Make this setup script more robust (plus conda recipe)
setup(
    ext_modules=[
        CUDAExtension(
            name="zipfft.zipfft_binding",
            sources=[
                "src/cuda/zipfft_binding.cu",
                "src/cuda/complex_fft_1d.cu",
                "src/cuda/real_fft_1d.cu",
                "src/cuda/padded_real_fft_1d.cu",
                "src/cuda/padded_real_conv_1d.cu",
            ],
            include_dirs=[
                pybind11.get_include(),
            ],
            extra_compile_args={
                # "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",  # Undefine PyTorch default macros; Necessary to get commondx compiled
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    version=__version__,
)
