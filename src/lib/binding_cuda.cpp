// C++ file for binding function calls to Python
#include <stdio.h>
#include <pybind11/pybind11.h>
#include "fft_kernels.cuh"


// Binding for the FFT launcher
int run_fft() {
    return simple_block_fft_1D_r2c_launcher();
}

PYBIND11_MODULE(binding_cuda, m) {
    m.doc() = "pybind11 binding example";
    m.def("run_fft", &run_fft, "Run the simple_block_fft_1D_r2c_launcher FFT kernel");
}