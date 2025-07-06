These are auto-generated .inc files for the pre-defined FFT shapes & data types since cuFFTDx needs to know the execution shapes and types at compile time.
Auto-generation is used in-place of manually writing implementation code and switch statements so users can easily configure what FFT/Conv types should be compiled and to build into auto-tuning frameworks.

* Extension `_implementations.inc` are for CUDA template implementations
* Extension `_asserts.inc` are static assertions for CUDA files
* Extension `_binding_cases.inc` are switch case statements for the Python bindings