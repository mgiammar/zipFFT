These are auto-generated .inc files for the pre-defined FFT shapes & data types since cuFFTDx needs to know the execution shapes and types at compile time. See [generate_fft_configs.py](../../generate_fft_configs.py) and the associated config JSON files to see how these files are built.

* Extension `_impl.inc` are for CUDA template implementations
* Extension `_assert.inc` are static assertions for CUDA files
* Extension `_bindings.inc` are switch case statements for the Python bindings