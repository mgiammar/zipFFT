"""Setup script for binding C++ code with Python using pybind11."""

from setuptools import setup, Extension
import pybind11

# TODO: Make this setup script more robust
setup(
    ext_modules=[
        Extension(
            "zipfft.binding",
            ["src/lib/binding.cpp"],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=["-O3"],  # Optimization level
        )
    ]
)
