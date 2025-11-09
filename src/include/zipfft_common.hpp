#ifndef ZIPFFT_COMMON_HPP_
#define ZIPFFT_COMMON_HPP_

// TODO: Remove unused includes in the future
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

// #include <cuda_runtime_api.h>
// #include <cufft.h>

#include <cufftdx.hpp>

#include "cuda_fp16.h"

#ifndef CUDA_CHECK_AND_EXIT
#define CUDA_CHECK_AND_EXIT(error)                                                        \
    {                                                                                     \
        auto status = static_cast<cudaError_t>(error);                                    \
        if (status != cudaSuccess) {                                                      \
            std::cout << cudaGetErrorString(status) << " " << __FILE__ << ":" << __LINE__ \
                      << std::endl;                                                       \
            std::exit(status);                                                            \
        }                                                                                 \
    }
#endif  // CUDA_CHECK_AND_EXIT

#ifndef CUFFT_CHECK_AND_EXIT
#define CUFFT_CHECK_AND_EXIT(error)                                                 \
    {                                                                               \
        auto status = static_cast<cufftResult>(error);                              \
        if (status != CUFFT_SUCCESS) {                                              \
            std::cout << status << " " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(status);                                                      \
        }                                                                           \
    }
#endif  // CUFFT_CHECK_AND_EXIT

namespace zipfft {
enum class dimension { x, y, z };

inline unsigned int get_cuda_device_arch() {
    int device;
    CUDA_CHECK_AND_EXIT(cudaGetDevice(&device));

    int major = 0;
    int minor = 0;
    CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));

    return static_cast<unsigned>(major) * 100 + static_cast<unsigned>(minor) * 10;
}

namespace detail {
template <typename, typename = void>
struct has_x_field : std::false_type {};

template <typename T>
struct has_x_field<T, std::void_t<decltype(T::x)>> : std::true_type {};

template <typename, typename = void>
struct has_y_field : std::false_type {};

template <typename T>
struct has_y_field<T, std::void_t<decltype(T::y)>> : std::true_type {};

// for mixed IO
template <typename T1, typename T1R, typename T2, typename T2R>
struct are_same {
    constexpr static bool value = std::is_same_v<T1, T1R> && std::is_same_v<T2, T2R>;
};

template <typename T1, typename T1R, typename T2, typename T2R>
inline constexpr bool are_same_v = are_same<T1, T1R, T2, T2R>::value;

template <typename T, typename = void>
struct has_value_type : std::false_type {};

template <typename T>
struct has_value_type<T, decltype((void)typename T::value_type(), void())> : std::true_type {};

template <typename T>
inline constexpr bool has_value_type_v = has_value_type<T>::value;

template <typename T, typename = void>
struct get_precision {
    using type = T;
};

template <typename T>
struct get_precision<T, std::enable_if_t<has_value_type_v<T>, void>> {
    using type = typename T::value_type;
};

}  // namespace detail

// This detects all variations of complex types:
// * cufftComplex
// * float2
// * cufftdx::complex<>
// useful for thrust transformations
template <typename T, typename = void>
struct has_complex_interface : std::false_type {};

template <typename T>
struct has_complex_interface<
    T, std::enable_if_t<detail::has_x_field<T>::value and detail::has_y_field<T>::value>>
    : std::is_same<decltype(T::x), decltype(T::y)> {};

template <typename T>
struct vector_type;

template <>
struct vector_type<float> {
    using type = float2;
};

template <>
struct vector_type<double> {
    using type = double2;
};

template <typename T, typename = void>
struct get_value_type {
    using type = T;
};

template <typename T>
struct get_value_type<T, std::void_t<typename T::value_type>> {
    using type = typename T::value_type;
};

template <typename T>
using get_value_type_t = typename get_value_type<T>::type;

template <typename T>
using value_type_t = typename T::value_type;

template <typename VecT>
struct get_scalar_component;

template <>
struct get_scalar_component<double2> {
    using type = double;
};

template <>
struct get_scalar_component<float2> {
    using type = float;
};

template <>
struct get_scalar_component<__half2> {
    using type = __half;
};

template <typename VecT>
using get_scalar_component_t = typename get_scalar_component<VecT>::type;

// Conversion utility for mapping between different floating-point types.
template <typename TargetPrecision, typename SourcePrecision>
__host__ __device__ constexpr TargetPrecision convert_scalar(const SourcePrecision& v) {
    using TP = TargetPrecision;
    using SP = SourcePrecision;

    TargetPrecision ret{};

    if constexpr (std::is_same_v<TP, SP>) {
        ret = v;
    } else if constexpr (detail::are_same_v<TP, float, SP, __half>) {
        ret = __half2float(v);
    } else if constexpr (detail::are_same_v<TP, __half, SP, float>) {
        ret = __float2half(v);
    } else if constexpr (detail::are_same_v<TP, float, SP, __nv_bfloat16>) {
        ret = __bfloat162float(v);
    } else if constexpr (detail::are_same_v<TP, __nv_bfloat16, SP, float>) {
        ret = __float2bfloat16(v);
    } else {
        ret = static_cast<TP>(v);
    }

    return ret;
}

// Conversion utility for complex types (e.g., __half2 --> float2 or float --> double)
// Constructed so both scalar and complex types can be supported.
// NOTE: Does not support scalar -> complex or complex -> scalar conversion.
template <typename TargetTypeOrPrecision, typename SourceType>
__host__ __device__ constexpr auto convert(const SourceType& v) {
    constexpr bool is_source_complex = detail::has_value_type_v<SourceType>;
    using target_precision = typename detail::get_precision<TargetTypeOrPrecision>::type;
    using converted_type = std::conditional_t<is_source_complex, cufftdx::complex<target_precision>,
                                              TargetTypeOrPrecision>;

    converted_type ret{};

    if constexpr (is_source_complex) {
        ret = converted_type{convert_scalar<target_precision>(v.real()),
                             convert_scalar<target_precision>(v.imag())};
    } else {
        ret = converted_type{convert_scalar<target_precision>(v)};
    }

    return ret;
}

// Utility function to get the zero value of a type.
template <class T>
inline __device__ constexpr T get_zero() {
    // If type is not complex, need to also check if it is __half2 type
    // since not handled by complex interface function.
    if constexpr (not zipfft::has_complex_interface<T>::value) {
        if constexpr (std::is_same_v<T, __half2>) {
            return __half2{};
        } else {
            return 0.;
        }
    } else {  // Type is complex
        using value_type = decltype(T::x);
        return T{get_zero<value_type>(), get_zero<value_type>()};
    }
}
struct identity {
    template <typename T>
    __device__ __forceinline__ T operator()(const T& val) {
        return val;
    }
};

}  // namespace zipfft

#endif  // ZIPFFT_COMMON_HPP_