#include "common.hpp"

// NOTE: This has a fallback from SM_120 -> SM_90 since 1200 (hopper) does not
// want to compile right now

namespace dispatcher {

    inline unsigned int get_cuda_device_arch() {
        int device;
        CUDA_CHECK_AND_EXIT(cudaGetDevice(&device));

        int major = 0;
        int minor = 0;
        CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
        CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));

        return static_cast<unsigned>(major) * 100 + static_cast<unsigned>(minor) * 10;
    }

    inline unsigned int get_multiprocessor_count(int device) {
        int multiprocessor_count = 0;
        CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
        return multiprocessor_count;
    }

    inline unsigned int get_multiprocessor_count() {
        int device = 0;
        CUDA_CHECK_AND_EXIT(cudaGetDevice(&device));
        return get_multiprocessor_count(device);
    }

    /**
     * @brief Launcher for a templated CUDA kernel functor which takes no arguments.
     * Determines the CUDA device architecture and calls the appropriate specialization of the functor.
     * 
     * @return template<template<unsigned int> class Functor> 
     */
    template<template<unsigned int> class Functor>
    inline int sm_runner() {
        const auto cuda_device_arch = get_cuda_device_arch();

        switch (cuda_device_arch) {
            // case 700: Functor<700>()(); return 0;
            // case 720: Functor<720>()(); return 0;
            // case 750: Functor<750>()(); return 0;
            case 800: Functor<800>()(); return 0;
            case 860: Functor<860>()(); return 0;
            case 870: Functor<870>()(); return 0;
            case 890: Functor<890>()(); return 0;
            case 900: Functor<900>()(); return 0;
            case 1200: Functor<900>()(); return 0;
        }
        return 1; // Unsupported architecture
    }

    /**
     * @brief Launcher for a templated CUDA kernel functor which takes a pointer to data as an argument.
     * Determines the CUDA device architecture and calls the appropriate specialization of the functor.
     * The Functor is expected to be templated on <unsigned int Arch, typename T, unsigned int FFTSize>.
     * 
     * @tparam Functor The functor template: template<unsigned int, typename, unsigned int> class.
     * @tparam T_actual The actual data type for this invocation.
     * @tparam FFTSize_actual The actual data size for this invocation.
     * @tparam IsForwardFFT_actual Whether the functor is for a forward FFT (true) or inverse FFT (false).
     * @tparam elements_per_thread_actual Number of elements processed per thread.
     * @tparam FFTs_per_block_actual Number of FFTs processed per block.
     * @param data Pointer to the data to be processed, allocated on the device.
     * @return int On success, returns 0. On failure (unsupported architecture), returns 1.
     */
    template< template <
            unsigned int, // Arch
            typename,     // T_functor_type
            unsigned int, // FFTSize_functor_type
            bool,         // IsForwardFFT_functor_type
            unsigned int, // elements_per_thread_functor_type
            unsigned int> // FFTs_per_block_functor_type
        class        Functor,
        typename     T_actual,
        unsigned int FFTSize_actual,
        bool         IsForwardFFT_actual,
        unsigned int elements_per_thread_actual,
        unsigned int FFTs_per_block_actual >
    inline int sm_runner_inplace(T_actual* data) {
        const auto cuda_device_arch = get_cuda_device_arch(); // Assuming get_cuda_device_arch() is defined

        switch (cuda_device_arch) {
            case 800:  Functor<800, T_actual, FFTSize_actual, IsForwardFFT_actual, elements_per_thread_actual, FFTs_per_block_actual>()(data); return 0;
            case 860:  Functor<860, T_actual, FFTSize_actual, IsForwardFFT_actual, elements_per_thread_actual, FFTs_per_block_actual>()(data); return 0;
            case 870:  Functor<870, T_actual, FFTSize_actual, IsForwardFFT_actual, elements_per_thread_actual, FFTs_per_block_actual>()(data); return 0;
            case 890:  Functor<890, T_actual, FFTSize_actual, IsForwardFFT_actual, elements_per_thread_actual, FFTs_per_block_actual>()(data); return 0;
            case 900:  Functor<900, T_actual, FFTSize_actual, IsForwardFFT_actual, elements_per_thread_actual, FFTs_per_block_actual>()(data); return 0;
            case 1200: Functor<900, T_actual, FFTSize_actual, IsForwardFFT_actual, elements_per_thread_actual, FFTs_per_block_actual>()(data); return 0;
            // Add more architectures as needed
        }
        // Consider logging or returning a more specific error for unsupported architecture
        std::cerr << "Unsupported CUDA architecture: " << cuda_device_arch << std::endl;
        return 1;
    }

    /**
     * @brief Launcher for a templated CUDA kernel functor which takes a pointer to input and output data as arguments.
     * Determines the CUDA device architecture and calls the appropriate specialization of the functor.
     *
     * The Functor is expected to be templated on:
     * <unsigned int Arch, typename Input_T, typename Output_T, unsigned int FFTSize, bool IsForwardFFT, unsigned int elements_per_thread, unsigned int FFTs_per_block>
     *
     * @tparam Functor The functor template: template<unsigned int, typename, typename, unsigned int, bool, unsigned int, unsigned int> class.
     * @tparam Input_T_actual The actual input data type for this invocation.
     * @tparam Output_T_actual The actual output data type for this invocation.
     * @tparam FFTSize_actual The actual data size for this invocation.
     * @tparam IsForwardFFT_actual Whether the functor is for a forward FFT (true) or inverse FFT (false).
     * @tparam elements_per_thread_actual Number of elements processed per thread.
     * @tparam FFTs_per_block_actual Number of FFTs processed per block.
     * @param input Pointer to the input data to be processed, allocated on the device.
     * @param output Pointer to the output data to be processed, allocated on the device.
     * @return int On success, returns 0. On failure (unsupported architecture), returns 1.
     */
    template< template <
            unsigned int, // Arch
            typename,     // Input_T_functor_type
            typename,     // Output_T_functor_type
            unsigned int, // FFTSize_functor_type
            bool,         // IsForwardFFT_functor_type
            unsigned int, // elements_per_thread_functor_type
            unsigned int> // FFTs_per_block_functor_type
        class        Functor,
        typename     Input_T_actual,
        typename     Output_T_actual,
        unsigned int FFTSize_actual,
        bool         IsForwardFFT_actual,
        unsigned int elements_per_thread_actual,
        unsigned int FFTs_per_block_actual >
    inline int sm_runner_out_of_place(Input_T_actual* input, Output_T_actual* output) {
        const auto cuda_device_arch = get_cuda_device_arch();

        switch (cuda_device_arch) {
            case 800:  Functor<800, Input_T_actual, Output_T_actual, FFTSize_actual, IsForwardFFT_actual, elements_per_thread_actual, FFTs_per_block_actual>()(input, output); return 0;
            case 860:  Functor<860, Input_T_actual, Output_T_actual, FFTSize_actual, IsForwardFFT_actual, elements_per_thread_actual, FFTs_per_block_actual>()(input, output); return 0;
            case 870:  Functor<870, Input_T_actual, Output_T_actual, FFTSize_actual, IsForwardFFT_actual, elements_per_thread_actual, FFTs_per_block_actual>()(input, output); return 0;
            case 890:  Functor<890, Input_T_actual, Output_T_actual, FFTSize_actual, IsForwardFFT_actual, elements_per_thread_actual, FFTs_per_block_actual>()(input, output); return 0;
            case 900:  Functor<900, Input_T_actual, Output_T_actual, FFTSize_actual, IsForwardFFT_actual, elements_per_thread_actual, FFTs_per_block_actual>()(input, output); return 0;
            case 1200: Functor<900, Input_T_actual, Output_T_actual, FFTSize_actual, IsForwardFFT_actual, elements_per_thread_actual, FFTs_per_block_actual>()(input, output); return 0;
        }
        std::cerr << "Unsupported CUDA architecture: " << cuda_device_arch << std::endl;
        return 1; // Unsupported architecture
    }

    /**
     * @brief Launcher for a templated CUDA kernel functor for padded real/complex FFTs, which takes a pointer to input and output data as arguments.
     * Determines the CUDA device architecture and calls the appropriate specialization of the functor.
     *
     * The Functor is expected to be templated on:
     * <unsigned int Arch, typename Input_T, typename Output_T, unsigned int SignalLength, unsigned int FFTSize, bool IsForwardFFT, unsigned int elements_per_thread, unsigned int FFTs_per_block>
     *
     * @tparam Functor The functor template: template<unsigned int, typename, typename, unsigned int, unsigned int, bool, unsigned int, unsigned int> struct.
     * @tparam Input_T_actual The actual input data type for this invocation.
     * @tparam Output_T_actual The actual output data type for this invocation.
     * @tparam SignalLength_actual The actual signal length for this invocation.
     * @tparam FFTSize_actual The actual FFT size for this invocation.
     * @tparam IsForwardFFT_actual Whether the functor is for a forward FFT (true) or inverse FFT (false).
     * @tparam elements_per_thread_actual Number of elements processed per thread.
     * @tparam FFTs_per_block_actual Number of FFTs processed per block.
     * @param input Pointer to the input data to be processed, allocated on the device.
     * @param output Pointer to the output data to be processed, allocated on the device.
     * @return int On success, returns 0. On failure (unsupported architecture), returns 1.
     */
    template< template <
            unsigned int, // Arch
            typename,     // Input_T_functor_type
            typename,     // Output_T_functor_type
            unsigned int, // SignalLength_functor_type
            unsigned int, // FFTSize_functor_type
            bool,         // IsForwardFFT_functor_type
            unsigned int, // elements_per_thread_functor_type
            unsigned int  // FFTs_per_block_functor_type
        > class Functor,
        typename     Input_T_actual,
        typename     Output_T_actual,
        unsigned int SignalLength_actual,
        unsigned int FFTSize_actual,
        bool         IsForwardFFT_actual,
        unsigned int elements_per_thread_actual,
        unsigned int FFTs_per_block_actual >
    inline int sm_runner_padded_out_of_place(Input_T_actual* input, Output_T_actual* output) {
        const auto cuda_device_arch = get_cuda_device_arch();

        switch (cuda_device_arch) {
            case 800:  Functor<800, Input_T_actual, Output_T_actual, SignalLength_actual, FFTSize_actual, IsForwardFFT_actual, elements_per_thread_actual, FFTs_per_block_actual>()(input, output); return 0;
            case 860:  Functor<860, Input_T_actual, Output_T_actual, SignalLength_actual, FFTSize_actual, IsForwardFFT_actual, elements_per_thread_actual, FFTs_per_block_actual>()(input, output); return 0;
            case 870:  Functor<870, Input_T_actual, Output_T_actual, SignalLength_actual, FFTSize_actual, IsForwardFFT_actual, elements_per_thread_actual, FFTs_per_block_actual>()(input, output); return 0;
            case 890:  Functor<890, Input_T_actual, Output_T_actual, SignalLength_actual, FFTSize_actual, IsForwardFFT_actual, elements_per_thread_actual, FFTs_per_block_actual>()(input, output); return 0;
            case 900:  Functor<900, Input_T_actual, Output_T_actual, SignalLength_actual, FFTSize_actual, IsForwardFFT_actual, elements_per_thread_actual, FFTs_per_block_actual>()(input, output); return 0;
            case 1200: Functor<900, Input_T_actual, Output_T_actual, SignalLength_actual, FFTSize_actual, IsForwardFFT_actual, elements_per_thread_actual, FFTs_per_block_actual>()(input, output); return 0;
        }
        std::cerr << "Unsupported CUDA architecture: " << cuda_device_arch << std::endl;
        return 1; // Unsupported architecture
    }

    template< template <
            unsigned int,
            typename,
            typename,
            unsigned int,
            unsigned int,
            unsigned int,
            unsigned int
        > class Functor,
        typename     RealType_Actual,
        typename     ComplexType_Actual,
        unsigned int SignalLength_Actual,
        unsigned int FFTSize_Actual,
        unsigned int elements_per_thread_Actual,
        unsigned int FFTs_per_block_Actual>
    inline int sm_runner_padded_conv(RealType_Actual* input_data, RealType_Actual* output_data, ComplexType_Actual* filter_data) {
        const auto cuda_device_arch = get_cuda_device_arch();

        switch (cuda_device_arch) {
            case 800:  Functor<800, RealType_Actual, ComplexType_Actual, SignalLength_Actual, FFTSize_Actual, elements_per_thread_Actual, FFTs_per_block_Actual>()(input_data, output_data, filter_data); return 0;
            case 860:  Functor<860, RealType_Actual, ComplexType_Actual, SignalLength_Actual, FFTSize_Actual, elements_per_thread_Actual, FFTs_per_block_Actual>()(input_data, output_data, filter_data); return 0;
            case 870:  Functor<870, RealType_Actual, ComplexType_Actual, SignalLength_Actual, FFTSize_Actual, elements_per_thread_Actual, FFTs_per_block_Actual>()(input_data, output_data, filter_data); return 0;
            case 890:  Functor<890, RealType_Actual, ComplexType_Actual, SignalLength_Actual, FFTSize_Actual, elements_per_thread_Actual, FFTs_per_block_Actual>()(input_data, output_data, filter_data); return 0;
            case 900:  Functor<900, RealType_Actual, ComplexType_Actual, SignalLength_Actual, FFTSize_Actual, elements_per_thread_Actual, FFTs_per_block_Actual>()(input_data, output_data, filter_data); return 0;
            case 1200: Functor<900, RealType_Actual, ComplexType_Actual, SignalLength_Actual, FFTSize_Actual, elements_per_thread_Actual, FFTs_per_block_Actual>()(input_data, output_data, filter_data); return 0;
        }
        std::cerr << "Unsupported CUDA architecture: " << cuda_device_arch << std::endl;
        return 1; // Unsupported architecture
    }
}