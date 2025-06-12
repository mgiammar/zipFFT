
// SM architectures were pulled from here:
// https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
//
// Compile options should be available for targeting only specific architectures


#include "common.hpp"

namespace runner {
    template<template<unsigned int> class Functor, typename... Args>
    inline int sm_runner(Args&&... args) {
        const auto cuda_device_arch = example::get_cuda_device_arch();

        switch (cuda_device_arch) {
            case 700: Functor<700>()(std::forward<Args>(args)...); return 0;
            case 720: Functor<720>()(std::forward<Args>(args)...); return 0;
            case 750: Functor<750>()(std::forward<Args>(args)...); return 0;
            case 800: Functor<800>()(std::forward<Args>(args)...); return 0;
            case 860: Functor<860>()(std::forward<Args>(args)...); return 0;
            case 870: Functor<870>()(std::forward<Args>(args)...); return 0;
            case 890: Functor<890>()(std::forward<Args>(args)...); return 0;
            case 900: Functor<900>()(std::forward<Args>(args)...); return 0;
            case 910: Functor<910>()(std::forward<Args>(args)...); return 0;
            case 1000: Functor<1000>()(std::forward<Args>(args)...); return 0;
            case 1010: Functor<1100>()(std::forward<Args>(args)...); return 0;
            case 1200: Functor<1200>()(std::forward<Args>(args)...); return 0;
        }
        return 1; // Unsupported architecture
    }
}