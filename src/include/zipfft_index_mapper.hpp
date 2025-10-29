#ifndef ZIPFFT_INDEX_MAPPER_HPP_
#define ZIPFFT_INDEX_MAPPER_HPP_

namespace zipfft {

template <int Dim, int Stride>
struct int_pair {
    static constexpr int size = Dim;
    static constexpr int stride = Stride;

    __device__ __host__ __forceinline__ size_t operator()(int id) const {
        return id * Stride;
    }
};

// Forward declaration
template <typename... Ts>
struct index_mapper;

// Helper trait to get the size of dimension N in an index_mapper
template <size_t N, typename Mapper>
struct dim_size;

// Base case: dimension 0
template <typename FirstDim, typename... RestDims>
struct dim_size<0, index_mapper<FirstDim, RestDims...>> {
    static constexpr int value = FirstDim::size;
};

// Recursive case: dimension N > 0
template <size_t N, typename FirstDim, typename... RestDims>
struct dim_size<N, index_mapper<FirstDim, RestDims...>> {
    static constexpr int value = dim_size<N - 1, index_mapper<RestDims...>>::value;
};

// Helper variable template for convenience
template <size_t N, typename Mapper>
inline constexpr int dim_size_v = dim_size<N, Mapper>::value;

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
    static constexpr int num_dims = 1 + sizeof...(NextDims);

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