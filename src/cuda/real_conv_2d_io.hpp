#ifndef ZIPFFT_REAL_CONV_2D_IO_HPP
#define ZIPFFT_REAL_CONV_2D_IO_HPP

/**
 * @brief This file includes I/O abstractions and optimizations for the 2D real convolution
 * operations. The 'io_conv' namespace encapsulates the load/store functions for both the
 * contiguous dimension (X) and strided dimension (Y).
 *
 */

#include <cufftdx.hpp>

#include "../include/zipfft_block_io.hpp"
#include "../include/zipfft_common.hpp"
#include "../include/zipfft_padded_io.hpp"
#include "../include/zipfft_strided_padded_io.hpp"

namespace zipfft {

template <dimension Dim, bool Forward, int Batches, class FFTX_, class IFFTX_, class FFTY_,
          class IFFTY_, unsigned int SignalLengthX, unsigned int SignalLengthY,
          unsigned int FFTSizeX, unsigned int FFTSizeY>
struct io_conv {
    // Convolution happens in the Y dimension (C2C transform)
    static constexpr bool is_r2c_conv =
        (cufftdx::type_of<FFTX_>::value == cufftdx::fft_type::r2c and
         cufftdx::type_of<IFFTX_>::value == cufftdx::fft_type::c2r);
    static constexpr bool is_c2c_conv =
        (cufftdx::type_of<FFTX_>::value == cufftdx::fft_type::c2c and
         cufftdx::type_of<IFFTX_>::value == cufftdx::fft_type::c2c);

    static_assert(is_r2c_conv or is_c2c_conv);

    // Redefinition of names for easier use across file
    using FFTX = std::conditional_t<Forward, FFTX_, IFFTX_>;
    using FFTY = std::conditional_t<Forward, FFTY_, IFFTY_>;

    // Determining the complex value type for the Y dimension. X dimension
    // will output this complex type for R2C and consume for C2R transform.
    using value_type = typename FFTY::value_type;
    static_assert(std::is_same_v<value_type, typename FFTX::value_type>);

    // Since Y dimension is C2C transform, lengths never change and
    // FFT::input_length == FFT::output_length == size_of<FFT>::value
    static constexpr unsigned int fft_size_y = cufftdx::size_of<FFTY>::value;
    static constexpr unsigned int fft_size_x = cufftdx::size_of<FFTX>::value;

    // Sizes of the input signals (how many data elements are in the input data)
    static constexpr unsigned int signal_length_y = SignalLengthY;
    static constexpr unsigned int signal_length_x = SignalLengthX;

    // Sizes of the output signals (valid cross-correlation/convolution lengths)
    static constexpr unsigned int valid_length_y = fft_size_y - signal_length_y + 1;
    static constexpr unsigned int valid_length_x = fft_size_x - signal_length_x + 1;

    // Determine if each dimension is padded (most likely will always be padded)
    static constexpr bool is_y_padded = fft_size_y != signal_length_y;
    static constexpr bool is_x_padded = fft_size_x != signal_length_x;

    // This is a value which determines what length for X other dimensions see.
    // Is number of complex elements in output for a R2C transform.
    // When the RealFFTOptions for the R2C/C2R transforms use complex_layout::natural,
    // then this value is fft_size_x / 2 + 1. If complex_layout::packed is used,
    // then this value is fft_size_x / 2 (element 0 hold real part of first and last value).
    // For powers of 2, complex_layout::packed may be more desirable.
    static constexpr unsigned int x_dim = FFTX_::output_length;

    ////////////////////////////////////////
    /// Utility functions for store/load ///
    ////////////////////////////////////////

    template <class FFT, typename GmemType, typename RmemType, class LoadOp = zipfft::identity,
              int BatchOffset, int BlockOffset, bool IsPadded, int SignalLength>
    __device__ __forceinline__ void load_contiguous(const GmemType* gmem, RmemType* rmem,
                                                    [[maybe_unused]] LoadOp op = {}) {
        using input_t = typename FFT::input_type;
        using complex_type = typename FFT::value_type;

        constexpr auto inner_loop_limit = sizeof(input_t) / sizeof(GmemType);

        // Inital global memory index based on shape of (Batches, Y, X)
        // being launched with grid dimensions (Batches, FFTsPerBlock, 1) and FFTs split
        // across the block's Y dimension
        unsigned int gmem_index = (blockIdx.x * BlockOffset + blockIdx.y * BatchOffset) +
                                  (threadIdx.x * inner_loop_limit + threadIdx.y * SignalLength);

#pragma unroll
        for (unsigned int i = 0; i < FFT::input_ept; ++i) {
            for (unsigned int j = 0; j < inner_loop_limit; ++j) {
                unsigned int local_fft_element =
                    i * FFT::stride * inner_loop_limit + threadIdx.x * inner_loop_limit + j;
                // if (local_fft_element < FFT::input_length) {}
                if (local_fft_element < SignalLength) {
                    reinterpret_cast<GmemType*>(rmem)[i * inner_loop_limit + j] =
                        reinterpret_cast<const GmemType*>(gmem)[gmem_index + j];
                } else if (IsPadded) {
                    reinterpret_cast<GmemType*>(rmem)[i * inner_loop_limit + j] =
                        get_zero<typename complex_type::value_type>();  // Zero padding
                }
            }
            gmem_index += FFT::stride * inner_loop_limit;
        }
    }

    template <class FFT, typename GmemType, typename RmemType, class StoreOp = zipfft::identity,
              int BatchOffset, int BlockOffset, bool IsPadded, int ValidLength>
    __device__ __forceinline__ void store_contiguous(const RmemType* rmem, GmemType* gmem,
                                                     [[maybe_unused]] StoreOp op = {}) {
        using output_t = typename FFT::output_type;
        using complex_type = typename FFT::value_type;

        constexpr auto inner_loop_limit = sizeof(output_t) / sizeof(GmemType);

        // Inital global memory index based on shape of (Batches, Y, X)
        // being launched with grid dimensions (Batches, FFTsPerBlock, 1) and FFTs split
        // across the block's Y dimension
        unsigned int gmem_index = (blockIdx.x * BlockOffset + blockIdx.y * BatchOffset) +
                                  (threadIdx.x * inner_loop_limit + threadIdx.y * ValidLength);

#pragma unroll
        for (unsigned int i = 0; i < FFT::output_ept; ++i) {
            for (unsigned int j = 0; j < inner_loop_limit; ++j) {
                unsigned int local_fft_element =
                    i * FFT::stride * inner_loop_limit + threadIdx.x * inner_loop_limit + j;

                // if (local_fft_element < FFT::output_length) {}
                if (local_fft_element < ValidLength) {
                    reinterpret_cast<GmemType*>(gmem)[gmem_index + j] =
                        reinterpret_cast<const GmemType*>(rmem)[i * inner_loop_limit + j];
                    // If we are outside of the valid length, skip write for truncation
                }
            }
            gmem_index += FFT::stride * inner_loop_limit;
        }
    }

    template <class FFT, typename GmemType, typename RmemType, class LoadOp = zipfft::identity,
              int BatchOffset, int BlockOffset, int Stride, bool IsPadded, int SignalLength>
    __device__ __forceinline__ void load_strided(const GmemType* gmem, RmemType* rmem,
                                                 [[maybe_unused]] LoadOp op = {}) {
        using input_t = typename FFT::input_type;

        static_assert(sizeof(input_t) == sizeof(GmemType),
                      "Strided load not implemented for non-matching types.");

        // Inital global memory index based on shape of (Batches, FFTSizeY, x_dim)
        // but where only SignalLengthY rows have meaningful values to read. Other
        // rows are zero-padded.
        // Launched with grid dimensions (Batches, FFTsPerBlock, 1).
        // Each block is reading in a column of data from the 2D array so
        // global memory indices are computed slightly differently than contiguous case.
        unsigned int gmem_index = (blockIdx.x * BlockOffset + blockIdx.y * BatchOffset) +
                                  (threadIdx.x * Stride + threadIdx.y);

#pragma unroll
        for (unsigned int i = 0; i < FFT::input_ept; ++i) {
            unsigned int local_fft_element = i * FFT::stride + threadIdx.x;

            if (local_fft_element < FFT::input_length) {
                if (local_fft_element < SignalLength) {
                    reinterpret_cast<input_t*>(rmem)[i] =
                        reinterpret_cast<const input_t*>(gmem)[gmem_index];
                } else if (IsPadded) {
                    reinterpret_cast<input_t*>(rmem)[i] = get_zero<input_t>();  // Zero padding
                }
                gmem_index += Stride * FFT::stride;
            }
        }
    }

    template <class FFT, typename GmemType, typename RmemType, class StoreOp = zipfft::identity,
              int BatchOffset, int BlockOffset, int Stride, bool IsPadded, int ValidLength>
    __device__ __forceinline__ void store_strided(const RmemType* rmem, GmemType* gmem,
                                                  [[maybe_unused]] StoreOp op = {}) {
        using output_t = typename FFT::output_type;

        static_assert(sizeof(output_t) == sizeof(GmemType),
                      "Strided store not implemented for non-matching types.");

        // Inital global memory index based on shape of (Batches, SignalLengthY, x_dim)
        // being launched with grid dimensions (Batches, FFTsPerBlock, 1)
        unsigned int gmem_index = (blockIdx.x * BlockOffset + blockIdx.y * BatchOffset) +
                                  (threadIdx.x * Stride + threadIdx.y);

#pragma unroll
        for (unsigned int i = 0; i < FFT::output_ept; ++i) {
            unsigned int local_fft_element = i * FFT::stride + threadIdx.x;

            if (local_fft_element < FFT::output_length) {
                if (local_fft_element < ValidLength) {
                    reinterpret_cast<output_t*>(gmem)[gmem_index] =
                        (reinterpret_cast<const output_t*>(rmem)[i]);
                }
                // If we are outside of the valid length, skip write for truncation
                gmem_index += Stride * FFT::stride;
            }
        }
    }

    //////////////////////////////////////////////////////
    /// Abstracted functions for load/store operations ///
    //////////////////////////////////////////////////////

    /**
     * @brief Load data from global memory to register memory. Automatically decides IO function
     * based on templated parameters (dimension, forward/inverse, padding, etc.).
     * Assumes that the kernel calling this function has been launched with grid dimensions
     * (FFTsPerBlock, Batches, 1) and there is no additional striding between batches.
     * Block dimensions match the FFT type descriptor from cuFFTDx.
     *
     * @tparam GmemType - Data type in global memory
     * @tparam RmemType - Data type of register memory
     * @tparam LoadOp - Load operation to apply upon load (default is identity)
     * @param gmem - Pointer to global memory space
     * @param rmem - Pointer to register memory space
     * @param op - Load operation to apply
     */
    template <typename GmemType, typename RmemType, class LoadOp = zipfft::identity>
    __device__ __forceinline__ void load_gmem_to_rmem(const GmemType* gmem, RmemType* rmem,
                                                      [[maybe_unused]] LoadOp op = {}) {
        // Along the strided dimension (Y)
        if constexpr (Dim == dimension::y) {
            constexpr bool is_load_padded = is_y_padded and Forward;

            // Determine the pointer offsets for accessing columns/batches based on assumed
            // shape of global memory array (Batches, SignalLengthY, x_dim). block_offset -
            // Amount of space between sequential blocks in the grid (blockIdx.x) batch_offset -
            // Amount of space between batches in the grid (blockIdx.y)
            // clang-format off
            constexpr int block_offset  = FFTY::ffts_per_block;
            constexpr int batch_offset  = (Forward) ? x_dim * fft_size_y : x_dim * fft_size_y;
            constexpr int stride        = (Forward) ? x_dim : x_dim;
            constexpr int signal_length = (Forward) ? signal_length_y : fft_size_y;
            // clang-format on

            load_strided<FFTY, GmemType, RmemType, LoadOp, batch_offset, block_offset, stride,
                         is_load_padded, signal_length>(gmem, rmem, op);
        } else {  // Along the contiguous dimension (X)
            constexpr bool is_load_padded = is_x_padded and Forward;

            // Determine the pointer offsets for accessing rows/batches based on assumed
            // shape of global memory array.
            //
            // When a forward FFT, the global shape is (Batches, signal_length_y, signal_length_x).
            // When this is an inverse FFT, the global shape is (Batches, y_dim, x_dim).
            // batch_offset - Amount of space between successive batches (blockIdx.y)
            // block_offset - Amount of space between sequential FFTs in the grid (blockIdx.x)
            // signal_length - Number of elements to read in (for zero-padding)
            // clang-format off
            constexpr int batch_offset  = (Forward) ? signal_length_x * signal_length_y : x_dim * fft_size_y;
            constexpr int block_offset  = (Forward) ? FFTX::ffts_per_block * signal_length_x : FFTX::ffts_per_block * x_dim;
            constexpr int signal_length = (Forward) ? signal_length_x : x_dim;
            // clang-format on

            load_contiguous<FFTX, GmemType, RmemType, LoadOp, batch_offset, block_offset,
                            is_load_padded, signal_length>(gmem, rmem, op);
        }
    }

    template <typename GmemType, typename RmemType, class StoreOp = zipfft::identity>
    __device__ __forceinline__ void store_rmem_to_gmem(GmemType* gmem, const RmemType* rmem,
                                                       [[maybe_unused]] StoreOp op = {}) {
        // Along the strided dimension (Y)
        if constexpr (Dim == dimension::y) {
            constexpr bool is_store_padded = is_y_padded and not Forward;

            // Determine the pointer offsets for accessing columns/batches based on assumed
            // shape of global memory array (Batches, SignalLengthY, x_dim). block_offset -
            // Amount of space between sequential blocks in the grid (blockIdx.x) batch_offset -
            // Amount of space between batches in the grid (blockIdx.y)
            // NOTE: workspace is same shape for input/output always
            // clang-format off
            constexpr int block_offset  = FFTY::ffts_per_block;
            constexpr int batch_offset  = (not Forward) ? x_dim * fft_size_y : x_dim * fft_size_y;
            constexpr int stride        = (not Forward) ? x_dim : x_dim;
            constexpr int valid_length  = (not Forward) ? valid_length_y : fft_size_y;
            // clang-format on

            store_strided<FFTY, GmemType, RmemType, StoreOp, batch_offset, block_offset, stride,
                          is_store_padded, valid_length>(rmem, gmem, op);
        } else {  // Along the contiguous dimension (X)
            constexpr bool is_store_padded = is_x_padded and not Forward;

            // Determine the pointer offsets for accessing rows/batches based on assumed
            // shape of global memory array (Batches, y_dim, SignalLengthX). block_offset -
            // Amount of space between sequential blocks in the grid (blockIdx.x) batch_offset -
            // Amount of space between batches in the grid (blockIdx.y)
            // clang-format off
            constexpr int batch_offset  = (Forward) ? x_dim * fft_size_y : valid_length_x * valid_length_y;
            constexpr int block_offset  = (Forward) ? FFTX::ffts_per_block * x_dim : FFTX::ffts_per_block * valid_length_x;
            constexpr int valid_length  = (Forward) ? x_dim : valid_length_x;
            // clang-format on

            store_contiguous<FFTX, GmemType, RmemType, StoreOp, batch_offset, block_offset,
                             is_store_padded, valid_length>(rmem, gmem, op);
        }
    }
};  // struct io_conv

}  // namespace zipfft

#endif  // ZIPFFT_REAL_CONV_2D_IO_HPP