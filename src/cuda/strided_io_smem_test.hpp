#ifndef ZIPFFT_IO_REAL_CONV_2D_HPP
#define ZIPFFT_IO_REAL_CONV_2D_HPP

#include <cufftdx.hpp>

#include "../include/zipfft_block_io.hpp"
#include "../include/zipfft_common.hpp"
#include "../include/zipfft_index_mapper.hpp"
#include "../include/zipfft_padded_io.hpp"

namespace zipfft {
// TODO: documentation
template <dimension Dim, bool Forward, int Batches, class FFTX_, class IFFTX_, class FFTY_,
          class IFFTY_, unsigned int SignalLengthX,
          unsigned int SignalLengthY>  // clang-format: off
class io_conv_smem {
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

    static constexpr auto num_shared_banks = (32 * sizeof(float)) / sizeof(value_type);

    // Since Y dimension is C2C transform, lengths never change and
    // FFT::input_length == FFT::output_length == size_of<FFT>::value
    static constexpr unsigned int fft_size_y = cufftdx::size_of<FFTY>::value;
    static constexpr unsigned int fft_size_x = cufftdx::size_of<FFTX>::value;

    // Sizes of the input signals
    static constexpr unsigned int signal_length_y = SignalLengthY;
    static constexpr unsigned int signal_length_x = SignalLengthX;

    // Sizes of the output signals (valid cross-correlation/convolution lengths)
    static constexpr unsigned int valid_length_y = fft_size_y - signal_length_y + 1;
    static constexpr unsigned int valid_length_x = fft_size_x - signal_length_x + 1;

    // Determine if each dimension is padded (most likely will always be padded)
    static constexpr bool is_y_padded = fft_size_y != signal_length_y;
    static constexpr bool is_x_padded = fft_size_x != signal_length_x;

    // This is a value which determines what length for X other dimensions see;
    // Is number of complex elements in output for a R2C transform
    static constexpr unsigned int x_dim = FFTX_::output_length;

    // For 2D: Layout is (Batch=Z, Y, X) with X contiguous
    static constexpr unsigned int flat_batch_size = signal_length_y * x_dim;
    static constexpr unsigned int flat_signal_size = signal_length_y * signal_length_x;

    // In this loading scheme the global memory accesses are coalesced and shared
    // memory bank conflicts are minimized by padding. This is achieved by:
    // 1. Contiguous threads loading corresponding elements from subsequent batches
    //      and not from the same batch, as would take place normally. This emulates
    //      switching blockDim.x with blockDim.y, because stride between batches is 1
    //      and stride between elements from the same batch is the total number of
    //      subbatches.
    // 2. Since the stores to shared memory will be performed strided by a high power
    //      of 2 it is necessary to pad them by the number of threads which will be
    //      performing this. Hence the _pad values are created based on fpb and warp
    //      size. This padding works well only for powers of 2 (which is a fair
    //      case for the image/template sizes zipFFT is trying to target).
    static constexpr unsigned int warp_size = 32;
    static constexpr auto y_fpb = FFTY::ffts_per_block;
    static constexpr auto y_bank_offset =
        (signal_length_y % num_shared_banks == 0) ? (warp_size + (y_fpb - 1)) / y_fpb : 0;

    // // This layout defines the offsets in global memory to get a specific element.
    // // It can be addressed using only a single integer, which defines the global
    // // index of an element, which it maps to an appropriate offset in memory.
    // // Single index is decayed to full N-D index by applying modulo and division
    // // recursively. The pairs given in index_mapper definition are (Size, Stride) pairs.
    // // TODO: complete here

    // // // Y dimension subbatches (columns of X within same Y row)
    // // using global_layout_y_subbatches =
    // //     index_mapper<int_pair<x_dim, 1>>;  // X columns are contiguous in memory
    // // ;                                      // ???? Unsure if this is correct

    // Y layout for the intermediary stage between the R2C and C2C transforms.
    // The R2C kernel is only launched along the first signal_length_y rows, but
    // the workspace has the full fft_size_y rows. Array has shape (Batches, signal_length_y, x_dim)
    // after the R2C transform and is then transposed into (Batches, x_dim, signal_length_y)]
    // for contiguous access along the Y dimension in the C2C transform.
    // using global_layout_y_intermediate_r2c =
    //     index_mapper<int_pair<x_dim, 1>,                           // X is contiguous
    //                  int_pair<signal_length_y, x_dim>,             // Y elements strided by x_dim
    //                  int_pair<Batches, x_dim * signal_length_y>>;  // Batch offset
    using global_layout_y_intermediate_r2c =
        index_mapper<int_pair<signal_length_y, 1>,            // Y is contiguous
                     int_pair<x_dim, signal_length_y>,        // X elements strided
                     int_pair<Batches, x_dim * fft_size_y>>;  // Batch offset

    // Y layout for the intermediary stage between the C2C and C2R transforms.
    // The fuzed C2C kernel only writes values to the first (fft_size_y - signal_length_y + 1)
    // rows in a contiguous manner. A transposition is then applied between the C2C and C2R stages
    // to rearrange the data into (Batches, signal_length_y, x_dim).
    // using global_layout_y_intermediate_c2r =
    //     index_mapper<int_pair<x_dim, 1>,                      // X is contiguous
    //                  int_pair<valid_length_y, x_dim>,         // Y elements strided
    //                  int_pair<Batches, x_dim * fft_size_y>>;  // Batch offset
    using global_layout_y_intermediate_c2r =
        index_mapper<int_pair<valid_length_y, 1>,             // Y is contiguous
                     int_pair<x_dim, valid_length_y>,         // X elements strided
                     int_pair<Batches, x_dim * fft_size_y>>;  // Batch offset

    // // Full Y layout
    // using global_layout_y =
    //     index_mapper<int_pair<x_dim, 1>,                   // X is contiguous
    //                  int_pair<fft_size_y, x_dim>,          // Y elements strided by x_dim
    //                  int_pair<Batches, flat_batch_size>>;  // Batch offset

    // Pad shared memory in subbatch dimension to reduce shared memory bank conflicts
    using shared_layout_y = index_mapper<int_pair<signal_length_y, 1>,  // contiguous in smem
                                         int_pair<y_fpb, signal_length_y + y_bank_offset>>;
    // using shared_layout_y_transposed = index_mapper<int_pair<y_fpb, 1>,
    //                                                 int_pair<signal_length_y, y_fpb>>;

    static constexpr auto y_bank_offset_bytes =
        (signal_length_y + y_bank_offset) * y_fpb * sizeof(value_type);
    static constexpr auto y_shared_bytes =
        std::max<int>(FFTY::shared_memory_size, y_bank_offset_bytes);

    // These IO functions perform shared <---> register transfers based on provided layouts.
    // This also takes signal padding into acount.
    template <class FFT, int Subbatches, int SignalLength, class GlobalLayout, class SharedLayout>
    __device__ __forceinline__ void load_gmem_to_smem(const value_type* gmem,
                                                      value_type* smem) const {
        GlobalLayout global_layout;
        SharedLayout shared_layout;

        constexpr auto fpb = FFT::ffts_per_block;
        constexpr auto is_padded = SignalLength != cufftdx::size_of<FFT>::value;

        // While all blocks must be started with the same FPB and number of threads,
        // this example is flexible enough to allow batches % FPB != 0. In that case
        // the last block will process exactly (batches % FPB) subbatches, and
        // this_block_fpb is the actual value of computed FFTs for each block.
        const auto this_block_fpb = (blockIdx.x == Subbatches / fpb) ? Subbatches % fpb : fpb;

        // Load data from global by emulating a switch between
        // threadIdx.x and threadIdx.y
        const int tid = (threadIdx.x + threadIdx.y * blockDim.x);
        const int rev_elem_start = tid / this_block_fpb;
        const int rev_batch_id = tid % this_block_fpb;

        using input_t = typename FFT::input_type;
        auto input_smem = reinterpret_cast<input_t*>(smem);
        auto input_gmem = reinterpret_cast<const input_t*>(gmem);

        // Since it's a strided kernel it requires staging the data through shared memory to
        // achieve high global memory coalescing on loads and stores.
#pragma unroll
        for (int i = 0; i < FFT::input_ept; ++i) {
            const auto rev_elem_id = rev_elem_start + i * FFT::stride;
            const auto global_rev_batch_id = rev_batch_id + blockIdx.x * fpb;
            if ((not FFT::requires_workspace and not is_padded) or (rev_elem_id < SignalLength)) {
                // /// DEBUGGING: Print the computed offset and values being loaded
                // const int global_offset =
                //     static_cast<int>(global_layout(rev_elem_id, global_rev_batch_id,
                //     blockIdx.y));
                // const int shared_offset =
                //     static_cast<int>(shared_layout(rev_elem_id, rev_batch_id));
                // if (threadIdx.x == 0) {
                //     printf("[LOAD_GMEM_TO_SMEM] blockIdx=(%u, %u, %u), threadIdx=(%u, %u), "
                //            "rev_elem_id=%u, rev_batch_id=%u, global_rev_batch_id=%u, "
                //            "global_offset=%d, shared_offset=%d, value=(%f, %f)\n",
                //            blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
                //            rev_elem_id, rev_batch_id, global_rev_batch_id, global_offset,
                //            shared_offset,
                //            //   static_cast<float>(
                //            //       reinterpret_cast<const
                //            float2*>(input_gmem)[global_offset].x),
                //            //   static_cast<float>(
                //            //       reinterpret_cast<const
                //            float2*>(input_gmem)[global_offset].y)); 0.0f, 0.0f);
                // }

                // input_smem[shared_layout(rev_elem_id, rev_batch_id)] =
                //     input_gmem[global_layout(rev_elem_id, global_rev_batch_id, blockIdx.y)];
            }
        }
    }

    template <class FFT, int Subbatches, int SignalLength, class SharedLayout, class GlobalLayout>
    __device__ __forceinline__ void store_smem_to_gmem(const value_type* smem,
                                                       value_type* gmem) const {
        GlobalLayout global_layout;
        SharedLayout shared_layout;

        constexpr auto is_padded = SignalLength != cufftdx::size_of<FFT>::value;
        constexpr auto fpb = FFT::ffts_per_block;
        const auto this_block_fpb = (blockIdx.x == Subbatches / fpb) ? Subbatches % fpb : fpb;

        // Load data from global by emulating a switch between
        // threadIdx.x and threadIdx.y
        const int tid = (threadIdx.x + threadIdx.y * blockDim.x);
        const int rev_elem_start = tid / this_block_fpb;
        const int rev_batch_id = tid % this_block_fpb;

        using output_t = typename FFT::output_type;
        auto output_gmem = reinterpret_cast<output_t*>(gmem);
        auto output_smem = reinterpret_cast<const output_t*>(smem);

#pragma unroll
        for (int i = 0; i < FFT::output_ept; ++i) {
            const auto rev_elem_id = rev_elem_start + i * FFT::stride;
            const auto global_rev_batch_id = rev_batch_id + blockIdx.x * fpb;
            if ((not FFT::requires_workspace and not is_padded) or (rev_elem_id < SignalLength)) {
                output_gmem[global_layout(rev_elem_id, global_rev_batch_id, blockIdx.y)] =
                    output_smem[shared_layout(rev_elem_id, rev_batch_id)];
            }
        }
    }

    // Shared memory must be synchronized before this function call, and no guarantees on
    // shared memory sync after function call
    template <class FFT, int SignalLength, class SharedLayout, class Op>
    __device__ __forceinline__ void load_smem_to_rmem(const value_type* smem,
                                                      value_type* rmem) const {
        SharedLayout shared_layout;
        Op op;

        static constexpr auto is_padded = SignalLength != cufftdx::size_of<FFT>::value;

        using input_t = typename FFT::input_type;
        auto input_rmem = reinterpret_cast<input_t*>(rmem);
        auto input_smem = reinterpret_cast<const input_t*>(smem);

#pragma unroll
        for (int i = 0; i < FFT::input_ept; ++i) {
            const auto elem_id = threadIdx.y + i * FFT::stride;
            const auto batch_id = threadIdx.x;

            // /// DEBUGGING: Print the computed offsets and values being loaded
            // const int shared_offset = static_cast<int>(shared_layout(elem_id, batch_id));
            // if (blockIdx.x == 0 && threadIdx.x == 0) {
            //     printf("[LOAD_SMEM_TO_RMEM] blockIdx=(%u, %u, %u), threadIdx=(%u, %u), elem_id=%u, "
            //            "batch_id=%u, shared_offset=%d, value=(%f, %f)\n",
            //            blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, elem_id,
            //            batch_id, shared_offset,
            //            static_cast<float>(
            //                reinterpret_cast<const float2*>(input_smem)[shared_offset].x),
            //            static_cast<float>(
            //                reinterpret_cast<const float2*>(input_smem)[shared_offset].y));
            // }

            // if ((not FFT::requires_workspace and not is_padded) or (elem_id < SignalLength)) {
            //     input_rmem[i] = op(input_smem[shared_layout(elem_id, batch_id)]);
            // } else if (is_padded and elem_id < FFT::input_length) {
            //     input_rmem[i] = get_zero<input_t>();  // Zero padding
            // }
        }
    }

    // Shared memory must be synchronized before
    // no guarantees on shared memory sync after
    template <class FFT, int SignalLength, class SharedLayout, class Op>
    __device__ __forceinline__ void store_rmem_to_smem(const value_type* rmem,
                                                       value_type* smem) const {
        SharedLayout shared_layout;
        Op op;

        static constexpr auto is_padded = SignalLength != cufftdx::size_of<FFT>::value;

        using output_t = typename FFT::output_type;
        auto output_smem = reinterpret_cast<output_t*>(smem);
        auto output_rmem = reinterpret_cast<const output_t*>(rmem);

#pragma unroll
        for (int i = 0; i < FFT::output_ept; ++i) {
            const auto elem_id = threadIdx.x + i * FFT::stride;
            const auto batch_id = threadIdx.y;
            if ((not FFT::requires_workspace and not is_padded) or (elem_id < SignalLength)) {
                output_smem[shared_layout(elem_id, batch_id)] = op(output_rmem[i]);
            }
        }
    }

public:
    static constexpr __device__ __host__ __forceinline__ size_t get_shared_bytes() {
        if (Dim == dimension::y) {
            return y_shared_bytes;
        } else {
            return FFTX::shared_memory_size;
        }
    }

    template <typename GmemType, typename SmemType, typename RmemType,
              class LoadOp = zipfft::identity>
    __device__ __forceinline__ void load_gmem_to_rmem(const GmemType* gmem,
                                                      [[maybe_unused]] SmemType* smem,
                                                      RmemType* rmem,
                                                      [[maybe_unused]] LoadOp op = {}) const {
        // C2C dimension (strided)
        if constexpr (Dim == dimension::y) {
            constexpr int y_batches = x_dim;
            /// NOTE: Since we've adapted the code to apply a transposition between the C2C and real
            /// kernels, we no longer need to stage into shared memory during the global loads
            // // clang-format off
            // load_gmem_to_smem<FFTY, y_batches, signal_length_y, global_layout_y_intermediate_r2c,
            // shared_layout_y>(gmem, smem);
            // __syncthreads();
            // load_smem_to_rmem<FFTY, signal_length_y, shared_layout_y, LoadOp>(smem, rmem);
            // // clang-format on
            constexpr bool is_load_padded = is_y_padded and Forward;
            using io_t =
                std::conditional_t<is_load_padded, zipfft::io_padded<FFTY, signal_length_y>,
                                   zipfft::io<FFTY>>;
            using input_t = std::conditional_t<is_load_padded, GmemType, typename FFTY::input_type>;

            // Determine additional pointer offsets for accessing different rows/batches of data
            // block_offset - Amount of space between two sequential batches
            // additional_offset - Isotropic offset added to all threads
            // constexpr int x_pad = fft_size_x - signal_length_x;
            constexpr int block_offset = (Forward) ? x_dim * signal_length_y : x_dim * fft_size_y;
            constexpr int additional_offset = 0;

            auto gmem_input = reinterpret_cast<const input_t*>(gmem);
            io_t::load(reinterpret_cast<const input_t*>(gmem + blockIdx.y * block_offset), rmem,
                       threadIdx.y + additional_offset, op);
        } else {  // X dimension (contiguous)
            constexpr bool is_load_padded = is_x_padded and Forward;
            using io_t =
                std::conditional_t<is_load_padded, zipfft::io_padded<FFTX, signal_length_x>,
                                   zipfft::io<FFTX>>;
            using input_t = std::conditional_t<is_load_padded, GmemType, typename FFTX::input_type>;

            // Determine additional pointer offsets for accessing different rows/batches of data
            // block_offset - Amount of space between two sequential batches
            // additional_offset - Isotropic offset added to all threads
            constexpr int block_offset =
                (Forward) ? signal_length_x * signal_length_y : x_dim * valid_length_y;
            constexpr int additional_offset = 0;

            // constexpr auto non_padded_block_offset =
            //     (Forward and is_r2c_conv) ? (signal_length_y * fft_size_x * FFTX::input_length)
            //                               : flat_batch_size;
            // constexpr auto block_offset = Forward ? flat_signal_size : non_padded_block_offset;

            // /// DEBUGGING: Print the different calculated offsets
            // if (blockIdx.x == 0 && threadIdx.x == 0 and threadIdx.y == 0) {
            //     printf("[LOAD X] blockIdx.x: %d, blockIdx.y: %d, additional_offset: %d, "
            //            "block_offset: %d\n",
            //            blockIdx.x, blockIdx.y, additional_offset, block_offset);
            // }

            auto gmem_input = reinterpret_cast<const input_t*>(gmem);
            io_t::load(reinterpret_cast<const input_t*>(gmem_input + blockIdx.y * block_offset),
                       rmem, threadIdx.y + additional_offset, op);
        }
    }

    template <typename RmemType, typename SmemType, typename GmemType,
              class StoreOp = zipfft::identity>
    __device__ __forceinline__ void store_rmem_to_gmem(const RmemType* rmem,
                                                       [[maybe_unused]] SmemType* smem,
                                                       GmemType* gmem,
                                                       [[maybe_unused]] StoreOp op = {}) const {
        // C2C dimension (strided)
        if constexpr (Dim == dimension::y) {
            /// NOTE: Since we've adapted the code to apply a transposition between the C2C and real
            /// kernels, we no longer need to stage into shared memory during the global loads
            // // clang-format off
            // constexpr int y_batches = x_dim;
            // store_rmem_to_smem<FFTY, signal_length_y, shared_layout_y, StoreOp>(rmem, smem);
            // __syncthreads();
            // store_smem_to_gmem<FFTY, y_batches, signal_length_y, shared_layout_y,
            // global_layout_y_intermediate_c2r>(smem, gmem);
            // // clang-format on
            constexpr bool is_store_padded = is_y_padded;
            using io_t =
                std::conditional_t<is_store_padded, zipfft::io_padded<FFTY, valid_length_y>,
                                   zipfft::io<FFTY>>;
            using output_t =
                std::conditional_t<is_store_padded, GmemType, typename FFTY::output_type>;

            // Determine additional pointer offsets for accessing different rows/batches of data
            // block_offset - Amount of space between two sequential batches
            // additional_offset - Isotropic offset added to all threads
            // constexpr int x_pad = fft_size_x - signal_length_x;
            constexpr int block_offset = (Forward) ? x_dim * fft_size_y : x_dim * valid_length_y;
            // constexpr int block_offset = (Forward) ? x_dim * fft_size_y : x_dim *
            // signal_length_y;
            constexpr int additional_offset = 0;

            // /// DEBUGGING: Print the calcualted offsets and store padded
            // if ((blockIdx.x == 0 || blockIdx.x == 1) && threadIdx.x == 0 and threadIdx.y == 0) {
            //     printf("[STORE Y] blockIdx.x: %d, blockIdx.y: %d, additional_offset: %d, "
            //            "block_offset: %d, is_store_padded: %d\n",
            //            blockIdx.x, blockIdx.y, additional_offset, block_offset, is_store_padded);
            // }

            auto gmem_output = reinterpret_cast<output_t*>(gmem);
            io_t::store(rmem, reinterpret_cast<output_t*>(gmem + blockIdx.y * block_offset),
                        threadIdx.y + additional_offset, op);
        } else {  // X dimension (contiguous)
            // For R2C, forward is padded loads, *contiguous stores*
            // For C2R, forward is contiguous loads, *padded stores*
            constexpr bool is_store_padded = is_x_padded and (not Forward);

            using io_t =
                std::conditional_t<is_store_padded, zipfft::io_padded<FFTX, valid_length_x>,
                                   zipfft::io<FFTX>>;
            using output_t =
                std::conditional_t<is_store_padded, GmemType, typename FFTX::output_type>;

            // Determine additional pointer offsets for accessing different rows/batches of data
            // block_offset - Amount of space between two sequential batches
            // additional_offset - Isotropic offset added to all threads
            // constexpr int y_pad = fft_size_y - signal_length_y;
            constexpr int block_offset =
                Forward ? x_dim * signal_length_y : valid_length_x * valid_length_y;
            constexpr int additional_offset = 0;

            // constexpr auto non_padded_block_offset =
            //     (Forward and is_r2c_conv) ? (signal_length_y * fft_size_x * FFTX::input_length)
            //                               : flat_batch_size;
            // constexpr auto block_offset = Forward ? flat_signal_size : non_padded_block_offset;

            // /// DEBUGGING: Print the different calculated offsets
            // if (blockIdx.x == 0 && threadIdx.x == 0 and threadIdx.y == 0) {
            //     printf("[STORE X] blockIdx.x: %d, blockIdx.y: %d, additional_offset: %d, "
            //            "block_offset: %d, is_store_padded=%d\n",
            //            blockIdx.x, blockIdx.y, additional_offset, block_offset, is_store_padded);
            // }

            auto gmem_output = reinterpret_cast<output_t*>(gmem);
            io_t::store(rmem, reinterpret_cast<output_t*>(gmem_output + blockIdx.y * block_offset),
                        threadIdx.y + additional_offset, op);
        }
    }

};  // class io_conv_smem
}  // namespace zipfft

#endif  // ZIPFFT_IO_REAL_CONV_2D_HPP