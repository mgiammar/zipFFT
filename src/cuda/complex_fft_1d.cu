#include <cuda_runtime.h>
#include <cufftdx.hpp>
#include <type_traits>

#include "../include/block_io.hpp"
#include "../include/common.hpp"
#include "../include/dispatcher.hpp"

// --- Kernel Definition ---
template<class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__
void block_fft_c2c_1d_kernel(typename FFT::value_type* data) {
    using complex_type = typename FFT::value_type;
    // Local array for thread
    complex_type thread_data[FFT::storage_size];
    const unsigned int local_fft_id = threadIdx.y;
    example::io<FFT>::load(data, thread_data, local_fft_id);

    // Execute the FFT with shared memory
    extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem);

    // Save results back to global memory
    example::io<FFT>::store(thread_data, data, local_fft_id);
}

// --- Launcher Definition ---
template<
    unsigned int Arch,
    typename     T,
    unsigned int FFTSize,
    bool         IsForwardFFT,
    unsigned int elements_per_thread,
    unsigned int FFTs_per_block>
inline void block_fft_c2c_1d_launcher(T* data) {
    using namespace cufftdx;

    // Since complex input to FFT, convert vector into its scalar type
    using scalar_precision_type = example::get_scalar_component_t<T>;
    constexpr auto fft_direction = IsForwardFFT ? fft_direction::forward : fft_direction::inverse;

    using FFT = decltype(
        Block() +
        Size<FFTSize>() +
        Type<fft_type::c2c>() +
        Direction<fft_direction>() +
        Precision<scalar_precision_type>() +
        ElementsPerThread<elements_per_thread>() +
        FFTsPerBlock<FFTs_per_block>() +
        SM<Arch>()
    );

    // Increase shared memory size, if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        block_fft_c2c_1d_kernel<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        FFT::shared_memory_size
    ));

    // Cast to cuFFTDx complex type form the FFT struct
    using complex_type = typename FFT::value_type;
    complex_type* data_t = reinterpret_cast<complex_type*>(data);

    // Launch the kernel and ensure no errors afterwards
    block_fft_c2c_1d_kernel<FFT><<<1, FFT::block_dim, FFT::shared_memory_size>>>(data_t);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
}

// --- Functor for Dispatcher ---
template<
    unsigned int Arch,
    typename     T_functor,
    unsigned int FFTSize_functor,
    bool         IsForwardFFT_functor,
    unsigned int elements_per_thread_functor,
    unsigned int FFTs_per_block_functor>
struct fft_dispatch_functor {
    void operator()(T_functor* data) {
        block_fft_c2c_1d_launcher<
            Arch,
            T_functor,
            FFTSize_functor,
            IsForwardFFT_functor,
            elements_per_thread_functor,
            FFTs_per_block_functor
        >(data);
    }
};


// --- Public API Function Template Definition ---
/**
 * @brief Callable function to perform a 1D complex-to-complex FFT using cuFFTDx.
 * 
 * @tparam T The data type (currently must be float2).
 * @tparam FFTSize Number of elements in the data array (must match supported FFT sizes).
 * @tparam IsForwardFFT Boolean indicating whether the FFT is forward (true) or inverse (false).
 * @tparam elements_per_thread Number of elements processed by each thread (default: 1).
 * @tparam FFTs_per_block Number of FFTs computed per thread block (default: 1).
 * @param data The pointer to the data array containing complex numbers, allocated on device.
 * @return int Upon successful execution, returns 0. Otherwise, returns an error code.
 */
template <
    typename     T,
    unsigned int FFTSize,
    bool         IsForwardFFT,
    unsigned int elements_per_thread,
    unsigned int FFTs_per_block>
int block_complex_fft_1d(T* data) {
    // Static assertions to ensure the correct types and sizes are used
    if constexpr (IsForwardFFT) {
#include "../generated/fwd_fft_c2c_1d_assertions.inc"
    } else {
#include "../generated/inv_fft_c2c_1d_assertions.inc"
    }

    // Call the dispatcher with the functor, passing template parameters
    int result = dispatcher::sm_runner_inplace<
        fft_dispatch_functor,
        T,
        FFTSize,
        IsForwardFFT,
        elements_per_thread,
        FFTs_per_block
    >(data);

    // Runtime assertion that the dispatcher returned successfully
    if (result != 0) {
        std::cerr << "block_complex_fft_1d: Error in dispatcher, result code: " << result << std::endl;
        std::exit(result);
    }

    return result;
}

// --- Template Instantiations ---
// Each of these instantiations corresponds to a pre-compiled version of the FFT
// for that data type and size since cuFFTDx needs to know the exact type and
// size of the FFT at compile time.
#include "../generated/fwd_fft_c2c_1d_implementations.inc"

#include "../generated/inv_fft_c2c_1d_implementations.inc"
