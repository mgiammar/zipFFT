#ifndef ZIPFFT_INDEX_MAPPER_HPP_
#define ZIPFFT_INDEX_MAPPER_HPP_

namespace zipfft {

template <int Dim, int Stride>
struct int_pair {
    static constexpr int size = Dim;

    __device__ __host__ __forceinline__ size_t operator()(int id) const {
        return id * Stride;
    }
};

template <typename... Ts>
struct index_mapper;

template <typename LastDim>
struct index_mapper<LastDim> {
    static constexpr int size = LastDim::size;

    __device__ __host__ __forceinline__ size_t operator()(int id) const {
        return LastDim{}(id);
    }
};

template <typename ThisDim, typename... NextDims>
struct index_mapper<ThisDim, NextDims...> {
    static constexpr int size = (ThisDim::size * ... * NextDims::size);

    // Flat coordinate addressing
    __device__ __host__ __forceinline__ size_t operator()(int id) const {
        constexpr int this_dim_size = ThisDim::size;
        return ThisDim{}(id % this_dim_size) + index_mapper<NextDims...>{}(id / this_dim_size);
    }

    // Natural coordinate addressing
    template <typename... Indices>
    __device__ __host__ __forceinline__ size_t operator()(int id, Indices... indices) const {
        static_assert(sizeof...(Indices) == sizeof...(NextDims));
        return ThisDim{}(id) + index_mapper<NextDims...>{}(indices...);
    }
};

}  // namespace zipfft

#endif  // ZIPFFT_INDEX_MAPPER_HPP_