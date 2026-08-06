#ifndef ZIPFFT_REAL_CONV_2D_IO_HPP
#define ZIPFFT_REAL_CONV_2D_IO_HPP

/**
 * @brief This file includes I/O abstractions and optimizations for the 2D real convolution
 * operations. The 'io_conv' namespace encapsulates the load/store functions for both the
 * contiguous dimension (X) and strided dimension (Y).
 *
 */

#include <cufftdx.hpp>

#include "../include/zipfft_common.hpp"

namespace zipfft {

template <dimension Dim, bool Forward, int Batches, class FFTX_, class IFFTX_, class FFTY_,
          class IFFTY_, unsigned int SignalLengthX, unsigned int SignalLengthY,
          unsigned int FFTSizeX, unsigned int FFTSizeY, bool UseTiledSwizzledIO = false>
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
                                                    LoadOp op = {}) {
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
                        op(reinterpret_cast<const GmemType*>(gmem)[gmem_index + j]);
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
                                                     StoreOp op = {}) {
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
                        op(reinterpret_cast<const GmemType*>(rmem)[i * inner_loop_limit + j]);
                    // If we are outside of the valid length, skip write for truncation
                }
            }
            gmem_index += FFT::stride * inner_loop_limit;
        }
    }

    template <class FFT, typename GmemType, typename RmemType, class LoadOp = zipfft::identity,
              int BatchOffset, int BlockOffset, int Stride, bool IsPadded, int SignalLength>
    __device__ __forceinline__ void load_strided(const GmemType* gmem, RmemType* rmem,
                                                 LoadOp op = {}) {
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
                        op(reinterpret_cast<const input_t*>(gmem)[gmem_index]);
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
                                                  StoreOp op = {}) {
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
                        op(reinterpret_cast<const output_t*>(rmem)[i]);
                }
                // If we are outside of the valid length, skip write for truncation
                gmem_index += Stride * FFT::stride;
            }
        }
    }

    // --- Tiled + swizzled strided load/store ---
    //
    // load_strided/store_strided above access global memory with a per-thread stride of
    // `Stride` (=x_dim) elements between consecutive threadIdx.x values, which for large x_dim
    // means each warp lane lands in its own, mostly-empty 32-byte sector (~25% sector
    // utilization; confirmed via ncu on the production kernel).
    //
    // Fix: stage each row-band (FFT::stride rows x FFT::ffts_per_block columns) through shared
    // memory. The piece that talks to *global* memory remaps thread->address so that
    // FFT::ffts_per_block contiguous columns (contiguous in the row-major array) are handled by
    // consecutive threads, filling each 32-byte sector completely. The piece that talks to
    // *registers* keeps the natural per-thread (row, col) ownership cuFFTDx expects.
    //
    // The natural-layout shared access has a constant per-thread stride of FFT::ffts_per_block
    // complex (8-byte) elements; decomposed into 32-bit bank accesses this causes an 8-way bank
    // conflict when FFT::ffts_per_block is a small power of two. Fixed by storing real/imag
    // components in separate (structure-of-arrays) float buffers, each padded to a row stride of
    // FFT::ffts_per_block+1 elements -- coprime with the 32 shared memory banks, which makes the
    // per-thread bank mapping a bijection (mathematically conflict-free), not just reduced.
    //
    // Requires FFT::ffts_per_block to be even (so that +1 padding is odd, hence coprime with the
    // power-of-two bank count); enforced via static_assert below. Callers on devices/configs
    // that cannot afford FFT::ffts_per_block >= 2 (e.g. insufficient shared memory at very large
    // FFT sizes) should instantiate io_conv with UseTiledSwizzledIO=false to fall back to the
    // plain strided path above -- the static_assert here only fires for instantiations that are
    // actually selected via `if constexpr`, so the fallback path never triggers it.
    //
    // `smem_scratch` must point to a caller-allocated shared memory region of at least
    // `2 * FFT::stride * (FFT::ffts_per_block + 1) * sizeof(float)` bytes, and must not be
    // concurrently in use by anything else while this function runs (see complex_conv_2d.cuh's
    // fused Y-kernel, which reuses the FFT's own execution shared memory buffer for this, since
    // the timing never overlaps).
    template <class FFT, typename GmemType, typename RmemType, class LoadOp = zipfft::identity,
              int BatchOffset, int BlockOffset, int Stride, bool IsPadded, int SignalLength>
    __device__ __forceinline__ void load_strided_tiled_swizzled(const GmemType* gmem,
                                                                RmemType* rmem,
                                                                GmemType* smem_scratch,
                                                                LoadOp op = {}) {
        using input_t = typename FFT::input_type;

        static_assert(sizeof(input_t) == sizeof(GmemType),
                      "Tiled strided load not implemented for non-matching types.");
        static_assert(FFT::ffts_per_block % 2 == 0,
                      "Tiled swizzled IO requires an even FFT::ffts_per_block (needed for the "
                      "coprime-padding bank-conflict-free guarantee).");

        constexpr unsigned int tile = FFT::ffts_per_block;
        constexpr unsigned int padded_tile = tile + 1;

        float* smem_re = reinterpret_cast<float*>(smem_scratch);
        float* smem_im = smem_re + FFT::stride * padded_tile;

        const unsigned int tx = threadIdx.x;
        const unsigned int ty = threadIdx.y;
        const unsigned int tid = ty * FFT::stride + tx;
        const unsigned int local_row = tid / tile;
        const unsigned int local_col = tid % tile;

        const unsigned int base_index = blockIdx.x * BlockOffset + blockIdx.y * BatchOffset;

#pragma unroll
        for (unsigned int i = 0; i < FFT::input_ept; ++i) {
            // Coalesced global READ using the remapped thread->address (fills whole sectors).
            const unsigned int global_row_remap = i * FFT::stride + local_row;
            if (global_row_remap < FFT::input_length) {
                const unsigned int gmem_index = base_index + global_row_remap * Stride + local_col;
                if (global_row_remap < SignalLength) {
                    input_t v = op(reinterpret_cast<const input_t*>(gmem)[gmem_index]);
                    smem_re[local_row * padded_tile + local_col] = reinterpret_cast<float*>(&v)[0];
                    smem_im[local_row * padded_tile + local_col] = reinterpret_cast<float*>(&v)[1];
                } else if (IsPadded) {
                    smem_re[local_row * padded_tile + local_col] = 0.0f;
                    smem_im[local_row * padded_tile + local_col] = 0.0f;
                }
            }
            __syncthreads();

            // Gather using the natural per-thread (row, col) ownership cuFFTDx expects.
            const unsigned int local_fft_element = i * FFT::stride + tx;
            if (local_fft_element < FFT::input_length) {
                input_t val;
                reinterpret_cast<float*>(&val)[0] = smem_re[tx * padded_tile + ty];
                reinterpret_cast<float*>(&val)[1] = smem_im[tx * padded_tile + ty];
                reinterpret_cast<input_t*>(rmem)[i] = val;
            }
            __syncthreads();
        }
    }

    template <class FFT, typename GmemType, typename RmemType, class StoreOp = zipfft::identity,
              int BatchOffset, int BlockOffset, int Stride, bool IsPadded, int ValidLength>
    __device__ __forceinline__ void store_strided_tiled_swizzled(const RmemType* rmem,
                                                                 GmemType* gmem,
                                                                 GmemType* smem_scratch,
                                                                 StoreOp op = {}) {
        using output_t = typename FFT::output_type;

        static_assert(sizeof(output_t) == sizeof(GmemType),
                      "Tiled strided store not implemented for non-matching types.");
        static_assert(FFT::ffts_per_block % 2 == 0,
                      "Tiled swizzled IO requires an even FFT::ffts_per_block (needed for the "
                      "coprime-padding bank-conflict-free guarantee).");

        constexpr unsigned int tile = FFT::ffts_per_block;
        constexpr unsigned int padded_tile = tile + 1;

        float* smem_re = reinterpret_cast<float*>(smem_scratch);
        float* smem_im = smem_re + FFT::stride * padded_tile;

        const unsigned int tx = threadIdx.x;
        const unsigned int ty = threadIdx.y;
        const unsigned int tid = ty * FFT::stride + tx;
        const unsigned int local_row = tid / tile;
        const unsigned int local_col = tid % tile;

        const unsigned int base_index = blockIdx.x * BlockOffset + blockIdx.y * BatchOffset;

#pragma unroll
        for (unsigned int i = 0; i < FFT::output_ept; ++i) {
            // Stage using the natural per-thread (row, col) ownership cuFFTDx produced.
            const unsigned int local_fft_element = i * FFT::stride + tx;
            if (local_fft_element < FFT::output_length) {
                output_t val = op(reinterpret_cast<const output_t*>(rmem)[i]);
                smem_re[tx * padded_tile + ty] = reinterpret_cast<float*>(&val)[0];
                smem_im[tx * padded_tile + ty] = reinterpret_cast<float*>(&val)[1];
            }
            __syncthreads();

            // Coalesced global WRITE using the remapped thread->address (fills whole sectors).
            const unsigned int global_row_remap = i * FFT::stride + local_row;
            if (global_row_remap < FFT::output_length && global_row_remap < ValidLength) {
                const unsigned int gmem_index = base_index + global_row_remap * Stride + local_col;
                output_t out;
                reinterpret_cast<float*>(&out)[0] = smem_re[local_row * padded_tile + local_col];
                reinterpret_cast<float*>(&out)[1] = smem_im[local_row * padded_tile + local_col];
                reinterpret_cast<output_t*>(gmem)[gmem_index] = out;
            }
            __syncthreads();
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
                                                      LoadOp op = {},
                                                      GmemType* smem_scratch = nullptr) {
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

            if constexpr (UseTiledSwizzledIO) {
                load_strided_tiled_swizzled<FFTY, GmemType, RmemType, LoadOp, batch_offset,
                                            block_offset, stride, is_load_padded, signal_length>(
                    gmem, rmem, smem_scratch, op);
            } else {
                load_strided<FFTY, GmemType, RmemType, LoadOp, batch_offset, block_offset, stride,
                             is_load_padded, signal_length>(gmem, rmem, op);
            }
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
                                                       StoreOp op = {},
                                                       GmemType* smem_scratch = nullptr) {
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

            // Do inverse FFT normalization (dividing by FFTSize) for reverse FFTs
            // to maintain consistency with cuFFT behavior
            if constexpr (!Forward) {
                zipfft::divide_by_scalar<float> norm_op(static_cast<float>(FFTSizeY));
                if constexpr (UseTiledSwizzledIO) {
                    store_strided_tiled_swizzled<
                        FFTY, GmemType, RmemType, zipfft::divide_by_scalar<float>, batch_offset,
                        block_offset, stride, is_store_padded, valid_length>(rmem, gmem,
                                                                             smem_scratch, norm_op);
                } else {
                    store_strided<FFTY, GmemType, RmemType, zipfft::divide_by_scalar<float>,
                                  batch_offset, block_offset, stride, is_store_padded,
                                  valid_length>(rmem, gmem, norm_op);
                }
            } else {
                if constexpr (UseTiledSwizzledIO) {
                    store_strided_tiled_swizzled<FFTY, GmemType, RmemType, StoreOp, batch_offset,
                                                 block_offset, stride, is_store_padded,
                                                 valid_length>(rmem, gmem, smem_scratch, op);
                } else {
                    store_strided<FFTY, GmemType, RmemType, StoreOp, batch_offset, block_offset,
                                  stride, is_store_padded, valid_length>(rmem, gmem, op);
                }
            }
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

            // Do inverse FFT normalization (dividing by FFTSize) for reverse FFTs
            // to maintain consistency with cuFFT behavior
            if constexpr (!Forward) {
                zipfft::divide_by_scalar<float> norm_op(static_cast<float>(FFTSizeX));
                store_contiguous<FFTX, GmemType, RmemType, zipfft::divide_by_scalar<float>,
                                 batch_offset, block_offset, is_store_padded, valid_length>(
                    rmem, gmem, norm_op);
            } else {
                store_contiguous<FFTX, GmemType, RmemType, StoreOp, batch_offset, block_offset,
                                 is_store_padded, valid_length>(rmem, gmem, op);
            }
        }
    }
};  // struct io_conv

}  // namespace zipfft

#endif  // ZIPFFT_REAL_CONV_2D_IO_HPP