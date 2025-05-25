// C++ file for binding function calls to Python
#include <stdio.h>
#include <pybind11/pybind11.h>


int add(int a, int b) {
    return a + b;
}

PYBIND11_MODULE(binding, m) {
    m.doc() = "pybind11 binding example"; // Optional module docstring
    m.def("add", &add, "A function that adds two numbers");
}