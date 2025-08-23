#pragma once

// Platform compatibility layer for libriscv
// Reduces code duplication across iOS, Android, macOS, Linux, Windows

#include <cstdint>

// Platform detection
#if defined(__APPLE__)
    #include <TargetConditionals.h>
    #if TARGET_OS_IPHONE
        #define LIBRISCV_PLATFORM_IOS 1
    #else
        #define LIBRISCV_PLATFORM_MACOS 1
    #endif
    #define LIBRISCV_PLATFORM_APPLE 1
#elif defined(__ANDROID__)
    #define LIBRISCV_PLATFORM_ANDROID 1
    #define LIBRISCV_PLATFORM_LINUX_LIKE 1
#elif defined(__linux__)
    #define LIBRISCV_PLATFORM_LINUX 1
    #define LIBRISCV_PLATFORM_LINUX_LIKE 1
#elif defined(_WIN32) || defined(_WIN64)
    #define LIBRISCV_PLATFORM_WINDOWS 1
#else
    #define LIBRISCV_PLATFORM_OTHER 1
#endif

// Platform capabilities
namespace riscv::platform {

// Event system capabilities
constexpr bool has_native_epoll() {
#ifdef LIBRISCV_PLATFORM_LINUX_LIKE
    return true;
#else
    return false;
#endif
}

constexpr bool has_kqueue() {
#ifdef LIBRISCV_PLATFORM_APPLE
    return true;
#else
    return false;
#endif
}

// File system capabilities  
constexpr bool has_getdents64() {
#ifdef LIBRISCV_PLATFORM_LINUX_LIKE
    return true;
#else
    return false;
#endif
}

constexpr bool has_dup3() {
#if defined(LIBRISCV_PLATFORM_LINUX_LIKE) || defined(__FreeBSD__) || defined(__OpenBSD__)
    return true;
#else
    return false;
#endif
}

constexpr bool has_pipe2() {
#ifdef LIBRISCV_PLATFORM_LINUX_LIKE
    return true;
#else
    return false; // Will emulate with pipe() + fcntl()
#endif
}

constexpr bool has_preadv() {
#if defined(LIBRISCV_PLATFORM_LINUX) && defined(SYS_preadv)
    return true;
#else
    return false; // Will emulate with individual pread() calls
#endif
}

// Random number generation
constexpr bool has_getrandom() {
#ifdef LIBRISCV_PLATFORM_LINUX
    return true;
#else
    return false; // Will use platform-specific alternatives
#endif
}

constexpr bool has_security_framework() {
#ifdef LIBRISCV_PLATFORM_MACOS
    return true;
#else
    return false;
#endif
}

// Time system capabilities
constexpr bool has_clock_gettime() {
#ifndef LIBRISCV_PLATFORM_APPLE
    return true;
#else
    return false; // Will emulate with mach_absolute_time
#endif
}

constexpr bool has_mach_time() {
#ifdef LIBRISCV_PLATFORM_APPLE
    return true;
#else
    return false;
#endif
}

} // namespace riscv::platform
