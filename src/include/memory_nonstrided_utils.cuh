#include <c10/util/complex.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cufftdx.hpp>

template<class FFT>
static inline __device__
void load_nonstrided(const float2* input, float2* thread_data, unsigned int  local_fft_id) {

    unsigned int global_fft_id = blockIdx.x * (FFT::ffts_per_block / FFT::implicit_type_batching) + local_fft_id;
    const unsigned int offset = FFT::input_length * global_fft_id;
    const unsigned int stride = FFT::stride;
    unsigned int       index  = offset + threadIdx.x;

    #pragma unroll
    for (unsigned int i = 0; i < FFT::input_ept; ++i) {
        if ((i * stride + threadIdx.x) < FFT::input_length) {
            thread_data[i] = input[index];
            index += stride;
        }
    }
}

template<class FFT>
static inline __device__
void store_nonstrided(const float2* thread_data, float2* output, unsigned int local_fft_id) {

    unsigned int global_fft_id =
        blockIdx.x * (FFT::ffts_per_block / FFT::implicit_type_batching) + local_fft_id;
    const unsigned int offset = FFT::output_length * global_fft_id;
    const unsigned int stride = FFT::stride;
    unsigned int       index  = offset + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < FFT::output_ept; ++i) {
        if ((i * stride + threadIdx.x) < FFT::output_length) {
            output[index] = thread_data[i];
            index += stride;
        }
    }
}

template<class FFT, int padding_ratio>
static inline __device__
void load_padded_layered(const float2* input,
            float2* thread_data,
            unsigned int  local_fft_id) {

    unsigned int active_layers = FFT::input_length / padding_ratio;
    unsigned int extra_layers = FFT::input_length - active_layers;

    unsigned int global_fft_id = blockIdx.x * (FFT::ffts_per_block / FFT::implicit_type_batching) + local_fft_id;
    global_fft_id = global_fft_id + extra_layers * (global_fft_id / active_layers); // skip extra layers
    const unsigned int offset = FFT::input_length * global_fft_id;
    
    const unsigned int stride = FFT::stride;
    unsigned int       index  = offset + threadIdx.x;
    for (unsigned int i = 0; i < FFT::input_ept; ++i) {
        unsigned int fft_index = i * stride + threadIdx.x;

        if (fft_index < FFT::input_length / padding_ratio) {
            thread_data[i] = input[index];
            index += stride;
        } else if (fft_index < FFT::input_length) {
            thread_data[i] = float2{0.0f, 0.0f};
            index += stride;
        }
    }
}

template<class FFT, int padding_ratio>
static inline __device__
void store_layered(const float2* thread_data,
            float2*             output,
            unsigned int        local_fft_id) {

    unsigned int active_layers = FFT::output_length / padding_ratio;
    unsigned int extra_layers = FFT::output_length - active_layers;
    
    unsigned int global_fft_id = blockIdx.x * (FFT::ffts_per_block / FFT::implicit_type_batching) + local_fft_id;
    global_fft_id = global_fft_id + extra_layers * (global_fft_id / active_layers); // skip extra layers

    const unsigned int offset = FFT::output_length * global_fft_id;
    const unsigned int stride = FFT::stride;
    unsigned int       index  = offset + threadIdx.x;

    for (int i = 0; i < FFT::output_ept; ++i) {
        if ((i * stride + threadIdx.x) < FFT::output_length) {
            output[index] = thread_data[i];
            index += stride;
        }
    }
}