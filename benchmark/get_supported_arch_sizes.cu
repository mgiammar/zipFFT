#include <array>
#include <cufftdx.hpp>
#include <iostream>
#include <utility>

template <unsigned int Size>
struct FFTDescriptor {
    using type = decltype(cufftdx::Size<Size>() + cufftdx::Type<cufftdx::fft_type::c2c>() +
                          cufftdx::Direction<cufftdx::fft_direction::forward>() + cufftdx::Block() +
                          cufftdx::Precision<float>());
};

template <unsigned int Start, unsigned int... Ns>
constexpr auto offset_sequence(std::integer_sequence<unsigned int, Ns...>) {
    return std::integer_sequence<unsigned int, (Start + Ns)...>{};
}

template <unsigned int Arch, unsigned int... Sizes>
constexpr auto make_support_table(std::integer_sequence<unsigned int, Sizes...>) {
    return std::array<bool, sizeof...(Sizes)>{
        cufftdx::is_supported<typename FFTDescriptor<Sizes>::type, Arch>::value...};
}

int main() {
    constexpr unsigned int kMinSize = 6223;
    constexpr unsigned int kMaxSize = 6223 + 1000;
    constexpr unsigned int kCount = kMaxSize - kMinSize + 1;
    using fft_sizes =
        decltype(offset_sequence<kMinSize>(std::make_integer_sequence<unsigned int, kCount>{}));

    // constexpr auto support_800 = make_support_table<800>(fft_sizes{});
    constexpr auto support_890 = make_support_table<890>(fft_sizes{});
    // constexpr auto support_900 = make_support_table<900>(fft_sizes{});

    std::cout << "size,sm800,sm890,sm900\n";
    for (std::size_t i = 0; i < support_890.size(); ++i) {
        const unsigned int fft_size = kMinSize + static_cast<unsigned int>(i);
        std::cout << fft_size
                  << ','
                  //   << support_800[i] << ','
                  << support_890[i] << '\n';
        //   << support_900[i] << '\n';
    }
}