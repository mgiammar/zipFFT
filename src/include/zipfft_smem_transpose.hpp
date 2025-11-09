#ifndef ZIPFFT_SMEM_TRANSPOSE_HPP
#define ZIPFFT_SMEM_TRANSPOSE_HPP

#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <type_traits>

#include "cuda_bf16.h"
#include "cuda_fp16.h"

namespace zipfft {

// Forward declaration of kernel
template <typename T, class LayoutSrc, class LayoutDst, class SharedMemoryLayoutSrc,
          class SharedMemoryLayoutDst, class ThreadLayoutSrc, class ThreadLayoutDst>
__global__ void batched_transpose_kernel(T const* input, T* output, int height, int width,
                                         LayoutSrc layout_src, LayoutDst layout_dst,
                                         SharedMemoryLayoutSrc, SharedMemoryLayoutDst,
                                         ThreadLayoutSrc, ThreadLayoutDst);

/**
 * @brief Batched 3D transpose using shared memory with padding for bank conflict avoidance
 *
 * Transforms data from (Batches, Height, Width) to (Batches, Width, Height) layout.
 * Uses CuTe library for layout management and optimization.
 * Based on NVIDIA CUTLASS library matrix transpose reference implementation.
 *
 * @tparam TileSizeX Tile width (default 64, must be divisible by ThreadBlockSizeX)
 * @tparam TileSizeY Tile height (default 32, must be divisible by ThreadBlockSizeY)
 * @tparam ThreadBlockSizeX Thread block width (default 32)
 * @tparam ThreadBlockSizeY Thread block height (default 8, total threads = 256)
 */
template <int TileSizeX = 64, int TileSizeY = 32, int ThreadBlockSizeX = 32,
          int ThreadBlockSizeY = 8>
struct BatchedTranspose {
    // Compile-time validation
    static_assert(TileSizeX % ThreadBlockSizeX == 0,
                  "TileSizeX must be divisible by ThreadBlockSizeX");
    static_assert(TileSizeY % ThreadBlockSizeY == 0,
                  "TileSizeY must be divisible by ThreadBlockSizeY");
    static_assert(ThreadBlockSizeX * ThreadBlockSizeY <= 1024,
                  "Total thread count must not exceed 1024");

    /**
     * @brief Launch batched transpose operation (internal implementation using double)
     */
    template <typename T>
    static cudaError_t launch_impl(T const* input, T* output, int batches, int height, int width,
                                   cudaStream_t stream) {
        // Validate input parameters
        if (input == nullptr || output == nullptr) {
            return cudaErrorInvalidValue;
        }
        if (batches <= 0 || height <= 0 || width <= 0) {
            return cudaErrorInvalidValue;
        }

        // For each batch, we need to process a (height x width) matrix
        // Create per-batch layouts
        auto const tensor_shape{cute::make_shape(height, width)};
        auto const tensor_shape_transposed{cute::make_shape(width, height)};

        // Input matrix: row-major height x width matrix per batch
        // Memory layout: [batch][height][width] with strides [height*width, width, 1]
        auto const global_memory_layout_src{
            cute::make_layout(tensor_shape, cute::GenRowMajor{})};  // (height, width) : (width, 1)

        // Output matrix: row-major width x height matrix per batch
        // Memory layout: [batch][width][height] with strides [width*height, height, 1]
        auto const global_memory_layout_dst{cute::make_layout(
            tensor_shape_transposed, cute::GenRowMajor{})};  // (width, height) : (height, 1)

        // Same output matrix, column-major view: height x width
        auto const global_memory_layout_dst_transposed{
            cute::make_layout(tensor_shape, cute::GenColMajor{})};  // (height, width) : (1, height)

        using TileSizeX_t = cute::Int<TileSizeX>;
        using TileSizeY_t = cute::Int<TileSizeY>;
        using TileSizeXPadded_t = cute::Int<TileSizeX + 1>;  // +1 for bank conflict avoidance

        constexpr auto block_shape{cute::make_shape(TileSizeY_t{}, TileSizeX_t{})};

        // Shared memory layouts with padding (following CUTLASS reference)
        auto const shared_memory_layout_src_padded{
            cute::make_layout(block_shape, cute::make_stride(TileSizeXPadded_t{}, cute::Int<1>{}))};

        using ThreadBlockSizeX_t = cute::Int<ThreadBlockSizeX>;
        using ThreadBlockSizeY_t = cute::Int<ThreadBlockSizeY>;

        constexpr auto thread_block_shape{
            cute::make_shape(ThreadBlockSizeY_t{}, ThreadBlockSizeX_t{})};
        constexpr auto thread_block_shape_transposed{
            cute::make_shape(ThreadBlockSizeX_t{}, ThreadBlockSizeY_t{})};
        constexpr auto thread_layout{cute::make_layout(thread_block_shape, cute::GenRowMajor{})};
        constexpr auto thread_layout_transposed{
            cute::make_layout(thread_block_shape_transposed, cute::GenColMajor{})};

        // Grid: tiles in x, tiles in y, batches in z
        // Calculate number of tiles needed
        int num_tiles_x = (width + TileSizeX - 1) / TileSizeX;
        int num_tiles_y = (height + TileSizeY - 1) / TileSizeY;

        dim3 const grid_dim{static_cast<unsigned int>(num_tiles_x),
                            static_cast<unsigned int>(num_tiles_y),
                            static_cast<unsigned int>(batches)};
        dim3 const thread_dim{ThreadBlockSizeX * ThreadBlockSizeY};

        // Launch kernel with batch index and per-batch layouts
        batched_transpose_kernel<<<grid_dim, thread_dim, 0, stream>>>(
            input, output, height, width, global_memory_layout_src,
            global_memory_layout_dst_transposed, shared_memory_layout_src_padded,
            shared_memory_layout_src_padded, thread_layout, thread_layout_transposed);

        return cudaGetLastError();
    }

    /**
     * @brief Launch batched transpose - uses double internally for all types
     */
    template <typename T>
    static cudaError_t launch(T const* input, T* output, int batches, int height, int width,
                              cudaStream_t stream = 0) {
        static_assert(sizeof(T) == sizeof(double) || sizeof(T) == sizeof(float),
                      "Only float and double-sized types supported");

        if constexpr (sizeof(T) == sizeof(double)) {
            // Use double for 8-byte types (float2, double)
            return launch_impl<double>(reinterpret_cast<double const*>(input),
                                       reinterpret_cast<double*>(output), batches, height, width,
                                       stream);
        } else {
            // Use float for 4-byte types
            return launch_impl<float>(reinterpret_cast<float const*>(input),
                                      reinterpret_cast<float*>(output), batches, height, width,
                                      stream);
        }
    }
};

// Kernel implementation (following CUTLASS reference style)
template <typename T, class LayoutSrc, class LayoutDst, class SharedMemoryLayoutSrc,
          class SharedMemoryLayoutDst, class ThreadLayoutSrc, class ThreadLayoutDst>
__global__ void batched_transpose_kernel(T const* input, T* output, int height, int width,
                                         LayoutSrc layout_src, LayoutDst layout_dst,
                                         SharedMemoryLayoutSrc, SharedMemoryLayoutDst,
                                         ThreadLayoutSrc, ThreadLayoutDst) {
    using Element = T;

    CUTE_STATIC_ASSERT_V(
        cute::size(SharedMemoryLayoutSrc{}) == cute::size(SharedMemoryLayoutDst{}),
        "SharedMemoryLayoutSrc and SharedMemoryLayoutDst must have the same size.");

    __shared__ Element shared_memory[cute::cosize(SharedMemoryLayoutSrc{})];

    auto tensor_cache_src{
        cute::make_tensor(cute::make_smem_ptr(shared_memory), SharedMemoryLayoutSrc{})};
    auto tensor_cache_dst{
        cute::make_tensor(cute::make_smem_ptr(shared_memory), SharedMemoryLayoutDst{})};

    // Calculate batch offset - each batch is (height * width) elements
    int batch_idx = blockIdx.z;
    int batch_offset_src = batch_idx * height * width;
    int batch_offset_dst = batch_idx * width * height;

    // Create tensors for this specific batch
    auto const tensor_src{
        cute::make_tensor(cute::make_gmem_ptr(input + batch_offset_src), layout_src)};
    auto const tensor_dst{
        cute::make_tensor(cute::make_gmem_ptr(output + batch_offset_dst), layout_dst)};

    // Tile the tensors
    constexpr auto block_shape = cute::shape(SharedMemoryLayoutSrc{});
    auto const tiled_tensor_src{cute::tiled_divide(tensor_src, block_shape)};
    auto const tiled_tensor_dst{cute::tiled_divide(tensor_dst, block_shape)};

    // Get tile for this block
    auto global_tile_src{
        tiled_tensor_src(cute::make_coord(cute::_, cute::_), blockIdx.y, blockIdx.x)};
    auto global_tile_dst{
        tiled_tensor_dst(cute::make_coord(cute::_, cute::_), blockIdx.y, blockIdx.x)};

    auto thread_global_tile_src{
        cute::local_partition(global_tile_src, ThreadLayoutSrc{}, threadIdx.x)};
    auto thread_global_tile_dst{
        cute::local_partition(global_tile_dst, ThreadLayoutDst{}, threadIdx.x)};

    auto thread_shared_tile_src{
        cute::local_partition(tensor_cache_src, ThreadLayoutSrc{}, threadIdx.x)};
    auto thread_shared_tile_dst{
        cute::local_partition(tensor_cache_dst, ThreadLayoutDst{}, threadIdx.x)};

    // Identity tensors for bounds checking
    auto const identity_tensor_src{cute::make_identity_tensor(
        cute::make_shape(cute::size<0>(global_tile_src), cute::size<1>(global_tile_src)))};
    auto const thread_identity_tensor_src{
        cute::local_partition(identity_tensor_src, ThreadLayoutSrc{}, threadIdx.x)};
    auto predicator_src{cute::make_tensor<bool>(cute::make_shape(
        cute::size<0>(thread_global_tile_src), cute::size<1>(thread_global_tile_src)))};

    auto const identity_tensor_dst{cute::make_identity_tensor(
        cute::make_shape(cute::size<0>(global_tile_dst), cute::size<1>(global_tile_dst)))};
    auto const thread_identity_tensor_dst{
        cute::local_partition(identity_tensor_dst, ThreadLayoutDst{}, threadIdx.x)};
    auto predicator_dst{cute::make_tensor<bool>(cute::make_shape(
        cute::size<0>(thread_global_tile_dst), cute::size<1>(thread_global_tile_dst)))};

    auto const num_max_columns{cute::stride<0>(global_tile_src)};
    auto const num_max_rows{cute::stride<1>(global_tile_dst)};
    constexpr auto global_tile_columns{cute::size<1>(global_tile_src)};
    constexpr auto global_tile_rows{cute::size<0>(global_tile_src)};

    // Compute predicates for source
    CUTE_UNROLL
    for (unsigned int i{0}; i < cute::size<0>(predicator_src); ++i) {
        CUTE_UNROLL
        for (unsigned int j{0}; j < cute::size<1>(predicator_src); ++j) {
            auto const thread_identity{thread_identity_tensor_src(i, j)};
            bool const is_row_in_bound{
                cute::get<0>(thread_identity) + blockIdx.y * global_tile_rows < num_max_rows};
            bool const is_column_in_bound{
                cute::get<1>(thread_identity) + blockIdx.x * global_tile_columns < num_max_columns};
            predicator_src(i, j) = is_row_in_bound && is_column_in_bound;
        }
    }

    // Compute predicates for destination
    CUTE_UNROLL
    for (unsigned int i{0}; i < cute::size<0>(predicator_dst); ++i) {
        CUTE_UNROLL
        for (unsigned int j{0}; j < cute::size<1>(predicator_dst); ++j) {
            auto const thread_identity{thread_identity_tensor_dst(i, j)};
            bool const is_row_in_bound{
                cute::get<0>(thread_identity) + blockIdx.y * global_tile_rows < num_max_rows};
            bool const is_column_in_bound{
                cute::get<1>(thread_identity) + blockIdx.x * global_tile_columns < num_max_columns};
            predicator_dst(i, j) = is_row_in_bound && is_column_in_bound;
        }
    }

    // Load from global to shared
    cute::copy_if(predicator_src, thread_global_tile_src, thread_shared_tile_src);
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();

    // Store from shared to global (transposed)
    cute::copy_if(predicator_dst, thread_shared_tile_dst, thread_global_tile_dst);
}

// Convenience function for float2 (uses double internally)
template <int TileSizeX = 64, int TileSizeY = 32, int ThreadBlockSizeX = 32,
          int ThreadBlockSizeY = 8>
cudaError_t transpose_batched_complex(float2 const* input, float2* output, int batches, int height,
                                      int width, cudaStream_t stream = 0) {
    return BatchedTranspose<TileSizeX, TileSizeY, ThreadBlockSizeX,
                            ThreadBlockSizeY>::template launch<float2>(input, output, batches,
                                                                       height, width, stream);
}

// Convenience function for float (real)
template <int TileSizeX = 64, int TileSizeY = 32, int ThreadBlockSizeX = 32,
          int ThreadBlockSizeY = 8>
cudaError_t transpose_batched_real(float const* input, float* output, int batches, int height,
                                   int width, cudaStream_t stream = 0) {
    return BatchedTranspose<TileSizeX, TileSizeY, ThreadBlockSizeX,
                            ThreadBlockSizeY>::template launch<float>(input, output, batches,
                                                                      height, width, stream);
}

}  // namespace zipfft

#endif  // ZIPFFT_SMEM_TRANSPOSE_HPP