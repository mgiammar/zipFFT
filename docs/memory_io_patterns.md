# Memory I/O Patterns and Implementations for zipFFT

The separable nature of multi-dimensional FFTs means that higher-dimensional FFTs can be built up from multiple 1D FFTs applied along different axes of some array.
This entails accessing memory in the linear array using different strides.
Image convolutions generally include some sequence of zero-padding applied to the input signal before performing the FFTs.
We will build upon these two basic access patterns (striding & padding) to implement efficient FFT-based image convolution algorithms.

## cuFFTDx API

The [cuFFTDx documentation](https://docs.nvidia.com/cuda/cufftdx/index.html) provides an comprehensive overview of the library and type descriptors.
Here, we are interested in executing relatively large FFTs and will focus on the [cuFFTDx block execution method](https://docs.nvidia.com/cuda/cufftdx/api/methods.html#block-execute-method) and build into I/O patterns with fit within the cuFFTDx API (partially for my own understanding).

What cuFFTDx provides is guarantees on 1D FFT executions within CUDA kernels _if and only if_ data is placed in/read from a particular layout.
Executing FFTs within a CUDA kernel can save on global memory trips if additional computations are happening before and/or after the FFTs.

### Block dimensions for execution

Each block FFT descriptor in cuFFTDx has a `block_dim` trait which is necessary when launching a kernel using that FFT descriptor.
The `block_dim` is calculated automatically once a FFT descriptor has been constructed, and the dimensions correspond to

- `block_dim.x = (size_of<FFT>::value / FFT::elements_per_thread)`: Number of threads working concurrently to compute a single FFT. Data elements are split among threads.
- `block_dim.y = (FFT::ffts_per_block / FFT::implicit_type_batching)`: Number of FFTs being calculated concurrently within the block.
- `block_dim.z = 1`: Unused.

<!-- squeezing multiple FFTs into the same kernel requires that they have the same block dims -->

### Outer batch dimension

Executing multiple FFTs in the same kernel launch (across the grid) requires an outer batch dimension when the number of FFTs exceeds `FFT::ffts_per_block`.
In the included NVIDIA code samples, the `grid_dim = (outer_batch, 1, 1)` where `blockIdx.x` corresponds to the FFT group being executed within the kernel.
zipFFT adopts the same convention for outer batch indexing.

### Implicit type batching in cuFFTDx (`__half` support)

When using reduced `__half` precision, cuFFTDx essentially executes two FFTs within the same memory.
The trait `FFT::implicit_type_batching` indicates how many values from different FFTs are batched together into a single memory element, and for `__half` execution this value is 2 (two `__half`-precision FFTs executed as 1 `float32`-precision FFT).
Otherwise `FFT::implicit_type_batching = 1`.

This introduces an additional adjustment when calculating the offset for the `batch_offset` variable: the number of FFTs per block must be divided by `FFT::implicit_type_batching` to properly account for the interleaved FFTs.

```c++
const unsigned int apparent_ffts_per_block = FFT::ffts_per_block / FFT::implicit_type_batching;
```

## Data layout and access in memory

Suppose we have a 2D array, `x`, with shape `(h, w)` which lies in contiguous global memory and indexed by

```
row 0:   (0,  1,    2,    3,    ...,  w-1 )
row 1:   (w,  w+1,  w+2,  w+3,  ...,  2w-1)
row 2:   (2w, 2w+1, 2w+2, 2w+3, ...,  3w-1)
...
row h-1: ((n-1)*w, (n-1)*w+1,   ..., n*w-1)
```

The data in `x` need loaded into `complex64` registers local to each thread (or stored back into global memory from thread registers) for execution, but `x` may not be of `complex64` type.

For real-to-complex transforms (and complex-to-real transforms), we assume `x` is of type `float32` or equivalent; type conversion is necessary in this case when data cannot be accessed sequentially and packed together.
For reduced precision FFTs (potentially reducing overall memory pressure), we assume a real-to-complex transform from `__half` type which again requires type conversions and data packing.

### General loop structure

Our data access loop, to an approximation, looks like

```c++
void load(const IOType* input, RegisterType* thread_data, ...) {
    // ...
    const unsigned int batch_offset = ...;
    const unsigned int stride       = ...;
    unsigned int       index        = ...;  // includes 'batch_offset'

    // iterate over all elements for this thread
    for (unsigned int i = 0; i < FFT::input_ept, i++) {
        if ((i * stride + threadIdx.x) < FFT::input_length) {  // bounds checking
            thread_data[i] = input[index];
        }
        index += stride
    }
}
```

with the store function looking extremely similar.
Type conversions here are ignored but will be revisited later.
The most important variables to define for different access patterns are `batch_offset`, `stride`, and `index`.
`batch_offset` corresponds to starting index offset for a group of threads computing the same FFT.
`stride` corresponds to the gap between sequential reads in memory.

#### Inner loop for type conversions

When the data types of `input` and `thread_data` are not type-compatible (i.e. `input` is not accessed contiguously or element packing structure is different), an additional inner loop is necessary to read/write data properly.
Here, `inner_loop_limit` controls how many of the global memory elements correspond to a single `thread_data` element.

```c++
constexpr auto inner_loop_limit = sizeof(input_t) / sizeof(IOType);
...

for (unsigned int i = 0; i < FFT::input_ept, i++) {
    if ((i * stride + threadIdx.x) < FFT::input_length) {  // bounds checking
        for (unsigned int j = 0; j < inner_loop_limit; j++) {
            thread_data[i * inner_loop_limit + j] = input[index + j];
        }
    }
    index += stride
}
```

### Contiguous FFT

For a contiguous 1D transform happening along the fastest (last) dimension of the array, data can be accessed sequentially for both the `input` data and `thread_data`.
The loop variables are therefore defined as

```c++
const unsigned int batch_offset = (blockIdx.x * apparent_ffts_per_block + threadIdx.y) * FFT::input_length;  // row first element idx
const unsigned int stride       = FFT::stride;                                                               // Stride between stored FFT elements
unsigned int       index        = batch_offset + threadIdx.x;                                                // Starting index for *this* thread
```

Assuming an FFT size of 16 with 4 elements per thread and 1 FFT per block, the memory access pattern for 4 threads in a block would look like

```
blockIdx    threadIdx    thread_data (index of input)

(0, 0, 0)   (0, 0, 0)    [0,  4,  8, 12]
(0, 0, 0)   (1, 0, 0)    [1,  5,  9, 13]
(0, 0, 0)   (2, 0, 0)    [2,  6, 10, 14]
(0, 0, 0)   (3, 0, 0)    [3,  7, 11, 15]
```

Applying this now to our 2D array, where `h=w=16`, and assuming again an FFT size of 16 with 4 elements per thread, 2 FFTs per block, our kernel indices would then look like

```
blockIdx    threadIdx    thread_data (index of input)

(0, 0, 0)   (0, 0, 0)    [0,  4,  8,  12]  # row 0
(0, 0, 0)   (1, 0, 0)    [1,  5,  9,  13]
(0, 0, 0)   (2, 0, 0)    [2,  6,  10, 14]
(0, 0, 0)   (3, 0, 0)    [3,  7,  11, 15]
(0, 0, 0)   (0, 1, 0)    [16, 20, 24, 28]  # row 1
(0, 0, 0)   (1, 1, 0)    [17, 21, 25, 29]
(0, 0, 0)   (2, 1, 0)    [18, 22, 26, 30]
(0, 0, 0)   (3, 1, 0)    [19, 23, 27, 31]

(1, 0, 0)  (0, 0, 0)     [32, 36, 40, 44]  # row 2
(1, 0, 0)  (0, 0, 0)     [33, 37, 41, 45]
...
```

#### Use of `reinterpret_cast` in contiguous access

Since data is access sequentially, the input and thread data can be accessed in the same pattern rather than needing non-contiguous reads/writes.
If the input data is real (e.g. `float`) but the register data is complex (e.g. `float2`) then the data can be `reinterpret_cast`-ed to `float*` since 

1. `sizeof(float2) == 2 * sizeof(float)`
2. cuFFTDx packs real input data as `c0 = {r0, r1}, c1 = {r2, r3}, ...` in memory.

_Note: real-to-complex and complex-to-real transforms will be most performant with powers-of-two sizes because of folded data packing._

### Padded FFT (contiguous access)

The major target of zipFT is zero-padded FFTs for image convolution where the input signal is padded with zeros from a constant `SignalLength` to `FFT::input_length`.
Padding happens within the loop by checking if the current thread write index exceeds `SignalLength` and placing a zero there if the condition is true.

The loop variables (for the contiguous access pattern, with zero-padding) are defined as

```c++
const unsigned int batch_offset = (blockIdx.x * apparent_ffts_per_block + threadIdx.y) * SignalLength;  // SignalLength, not FFT::input_length
const unsigned int stride       = FFT::stride;                                                          // Stride between stored FFT elements
unsigned int       index        = batch_offset + threadIdx.x * inner_loop_limit;                        // Starting index for *this* thread
```

Applying padded access to our 2D array example from before, with `h=w=16`, `SignalLength=16`, and `FFT::input_length=32`, our kernel indices would then look like

```
blockIdx    threadIdx    thread_data (index of input)

(0, 0, 0)   (0, 0, 0)    [0,  4,  8,  12, 0,  0,  0,  0]  # row 0
(0, 0, 0)   (1, 0, 0)    [1,  5,  9,  13, 0,  0,  0,  0]
(0, 0, 0)   (2, 0, 0)    [2,  6,  10, 14, 0,  0,  0,  0]
(0, 0, 0)   (3, 0, 0)    [3,  7,  11, 15, 0,  0,  0,  0]
(0, 0, 0)   (0, 1, 0)    [16, 20, 24, 28, 0,  0,  0,  0]  # row 1
(0, 0, 0)   (1, 1, 0)    [17, 21, 25, 29, 0,  0,  0,  0]
(0, 0, 0)   (2, 1, 0)    [18, 22, 26, 30, 0,  0,  0,  0]
(0, 0, 0)   (3, 1, 0)    [19, 23, 27, 31, 0,  0,  0,  0]

(1, 0, 0)  (0, 0, 0)     [32, 36, 40, 44, 0,  0,  0,  0]  # row 2
(1, 0, 0)  (0, 0, 0)     [33, 37, 41, 45, 0,  0,  0,  0]
...
```

### Strided FFT

For transforms happening along a slower dimension of the array, the stride between sequential reads/writes must be scaled by the stride of the array along that dimension.
Our starting indices also change to reflect accessing data not row-wise, but column-wise (assuming a 2D array or flattened last dimensions).

The strided access pattern loop variables are defined as 

```c++
const unsigned int batch_offset = blockIdx.x * apparent_ffts_per_block + threadIdx.y;        // column first element idx
const unsigned int stride       = FFT::stride * Stride;                                      // Include jump between rows
unsigned int       index        = batch_offset + (threadIdx.x * Stride * inner_loop_limit);  // Starting index for *this* thread
```

Applying strided access to our 2D array example from before, with `h=w=16`, and assuming again an FFT size of 16 with 4 elements per thread, 2 FFTs per block, our kernel indices would then look like

```
blockIdx    threadIdx    thread_data (index of input)

(0, 0, 0)   (0, 0, 0)    [0, 16, 32, 48]  # column 0
(0, 0, 0)   (1, 0, 0)    [1, 17, 33, 49]
(0, 0, 0)   (2, 0, 0)    [2, 18, 34, 50]
(0, 0, 0)   (3, 0, 0)    [3, 19, 35, 51]
(0, 0, 0)   (0, 1, 0)    [4, 20, 36, 52]  # column 1
(0, 0, 0)   (1, 1, 0)    [5, 21, 37, 53]
(0, 0, 0)   (2, 1, 0)    [6, 22, 38, 54]
(0, 0, 0)   (3, 1, 0)    [7, 23, 39, 55]

(1, 0, 0)  (0, 0, 0)     [8, 24, 40, 56]  # column 2
(1, 0, 0)  (1, 0, 0)     [9, 25, 41, 57]
...
```

#### Additional batch ID check

Reading in / writing out data from a strided kernel introduces an additional bounds check based on teh input array shape.
Consider the case where we have `w=13` columns in our data and our FFT structure computes 4 FFTs per block.
Block indices `blockIdx.x = {0, 1, 2}` will cover the transforms from columns 0 to 12 completely, but the last block launched, `blockIdx.x = 3`, only has 1 column to transform along.
If we read in / write out elements as normal without bounds checking on this last block, there will be an access violation when the loop tries to access columns 13, 14 and 15.

To prevent an access violation, an additional check on the `batch_offset` is necessary where we ensure `batch_offset < w` before reading/writing in the loop.

```c++
const unsigned int Batches = ...;  // total number of FFT batches (e.g. columns)
...
for (unsigned int i = 0; i < FFT::input_ept, i++) {
    if ((i * FFT::stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
        if (batch_offset < Batches) {  // additional bounds checking for strided access
            ... // Data access here
        }
        index += stride;
    }
}
```

### Strided + Padded FFT

Achieving a fully padded 2D FFT transform along both dimensions requires combining the strided and padded access patterns.
Again, zero-padding happens by checking if the current thread write index exceeds `SignalLength` conditionally placing a zero there.


The strided + padded access pattern loop variables roughly match those of the strided access pattern,

```c++
const unsigned int batch_offset = blockIdx.x * apparent_ffts_per_block + threadIdx.y;        // column first element idx
const unsigned int stride       = FFT::stride * Stride;                                      // Include jump between rows
unsigned int       index        = batch_offset + (threadIdx.x * Stride * inner_loop_limit);  // Starting index for *this* thread
```

Additionally, the bounds for checking the signal length limit must also be adjusted to account for striding.

```c++
const unsigned int signal_length_limit = SignalLength * Stride;
```

### Other considerations for I/O patterns

There are a handful of other data access patterns that zipFFT may want to revisit in the future which include:

- **Signal reversal**: The cross-correlation operation is equivalent to convolution with a reverse signal. It's possible that reversing the signal (during load) could improve throughput instead of calculating the complex conjugate during the point-wise multiplication step.
- **Pre-transposed data**: Maximizing coalesced memory access is critical for performance. Pre-transposing data so the longer, less-padded dimension lies along the fastest changing axis could improve memory throughput.
- **Shared memory I/O**: Using shared memory as a staging area for global memory reads/writes could improve performance for certain access patterns, especially strided access.

## I/O header API

To encapsulate the various I/O patterns described above, zipFFT provides a set of I/O header files which define load/store functions for different access patterns and data types.
Most of these are adapted from the NvidiaLibaraySamples repository with some necessary modifications and include

- `include/zipfft_block_io.hpp`: Contiguous access (no striding or padding).
- `include/zipfft_padded_io.hpp`: Contiguous access with zero-padding.
- `include/zipfft_strided_io.hpp`: Strided access (no padding).
- `include/zipfft_strided_padded_io.hpp`: Strided access with zero-padding.

Each of these headers have a templated structure under the `zipfft` namespace and inherit from the block `io` structure.
To promote interoperability, each structure has the `load` and `store` functions defined with the same function signatures.

```c++
// NOTE: excluding function qualifiers for brevity
template <typename RegisterType, typename IOType>
load(const IOType* input, RegisterType* thread_data, unsigned int local_fft_id);

template <typename RegisterType, typename IOType>
store(IOType* output, const RegisterType* thread_data, unsigned int local_fft_id);
```