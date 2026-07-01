/*
 * Copyright 2018 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#pragma once

// skcms_internals.h contains APIs shared by skcms' internals and its test tools.
// Please don't use this header from outside the skcms repo.

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ~~~~ General Helper Macros ~~~~
// skcms can leverage some C++ extensions when they are present.
#define ARRAY_COUNT(arr) (int)(sizeof((arr)) / sizeof(*(arr)))

#if defined(__clang__) && defined(__has_cpp_attribute)
    #if __has_cpp_attribute(clang::fallthrough)
        #define SKCMS_FALLTHROUGH [[clang::fallthrough]]
    #endif

    #ifndef SKCMS_HAS_MUSTTAIL
        // [[clang::musttail]] is great for performance, but it's not well supported and we run into
        // a variety of problems when we use it. Fortunately, it's an optional feature that doesn't
        // affect correctness, and usually the compiler will generate a tail-call even for us
        // whether or not we force it to do so.
        //
        // Known limitations:
        // - Sanitizers do not work well with [[clang::musttail]], and corrupt src/dst pointers.
        //   (https://github.com/llvm/llvm-project/issues/70849)
        // - Wasm tail-calls were only introduced in 2023 and aren't a mainstream feature yet.
        // - Clang 18 runs into an ICE on armv7/androideabi with [[clang::musttail]].
        //   (http://crbug.com/1504548)
        // - Android RISC-V also runs into an ICE (b/314692534)
        // - So does Linux ppc64le (https://github.com/llvm/llvm-project/issues/108014,
        //   https://github.com/llvm/llvm-project/issues/98859)
        // - LoongArch developers indicate they had to turn it off
        // - Windows builds generate incorrect code with [[clang::musttail]] and crash mysteriously.
        //   (http://crbug.com/1505442)
        #if __has_cpp_attribute(clang::musttail) && !__has_feature(memory_sanitizer) \
                                                 && !__has_feature(address_sanitizer) \
                                                 && !defined(__EMSCRIPTEN__) \
                                                 && !defined(__arm__) \
                                                 && !defined(__riscv) \
                                                 && !defined(__powerpc__) \
                                                 && !defined(__loongarch__) \
                                                 && !defined(_WIN32) && !defined(__SYMBIAN32__)
            #define SKCMS_HAS_MUSTTAIL 1
        #endif
    #endif
#endif

#ifndef SKCMS_FALLTHROUGH
    #define SKCMS_FALLTHROUGH
#endif
#ifndef SKCMS_HAS_MUSTTAIL
    #define SKCMS_HAS_MUSTTAIL 0
#endif

#if defined(__clang__)
    #define SKCMS_MAYBE_UNUSED __attribute__((unused))
    #pragma clang diagnostic ignored "-Wused-but-marked-unused"
#elif defined(__GNUC__)
    #define SKCMS_MAYBE_UNUSED __attribute__((unused))
#elif defined(_MSC_VER)
    #define SKCMS_MAYBE_UNUSED __pragma(warning(suppress:4100))
#else
    #define SKCMS_MAYBE_UNUSED
#endif

// sizeof(x) will return size_t, which is 32-bit on some machines and 64-bit on others.
// We have better testing on 64-bit machines, so force 32-bit machines to behave like 64-bit.
//
// Please do not use sizeof() directly, and size_t only when required.
// (We have no way of enforcing these requests...)
#define SAFE_SIZEOF(x) ((uint64_t)sizeof(x))

// Same sort of thing for _Layout structs with a variable sized array at the end (named "variable").
#define SAFE_FIXED_SIZE(type) ((uint64_t)offsetof(type, variable))

// If this isn't Clang, GCC, or Emscripten with SIMD support, we are in SKCMS_PORTABLE mode.
#if !defined(SKCMS_PORTABLE) && !(defined(__clang__) || \
                                  defined(__GNUC__) || \
                                  (defined(__EMSCRIPTEN__) && defined(__wasm_simd128__)))
    #define SKCMS_PORTABLE 1
#endif

// If we are in SKCMS_PORTABLE mode or running on a non-x86-64 platform, we can't enable HSW or SKX.
// We also disable HSW/SKX on Android, even if it's Android on x64, since it's unlikely to benefit.
#if defined(SKCMS_PORTABLE) || !defined(__x86_64__) || defined(ANDROID) || defined(__ANDROID__)
    #undef SKCMS_FORCE_HSW
    #if !defined(SKCMS_DISABLE_HSW)
        #define SKCMS_DISABLE_HSW 1
    #endif

    #undef SKCMS_FORCE_SKX
    #if !defined(SKCMS_DISABLE_SKX)
        #define SKCMS_DISABLE_SKX 1
    #endif
#endif

// ~~~~ Shared ~~~~
typedef struct skcms_ICCTag {
    uint32_t       signature;
    uint32_t       type;
    uint32_t       size;
    const uint8_t* buf;
} skcms_ICCTag;

typedef struct skcms_ICCProfile skcms_ICCProfile;
typedef struct skcms_TransferFunction skcms_TransferFunction;
typedef union skcms_Curve skcms_Curve;

void skcms_GetTagByIndex    (const skcms_ICCProfile*, uint32_t idx, skcms_ICCTag*);
bool skcms_GetTagBySignature(const skcms_ICCProfile*, uint32_t sig, skcms_ICCTag*);

float skcms_MaxRoundtripError(const skcms_Curve* curve, const skcms_TransferFunction* inv_tf);

// 252 of a random shuffle of all possible bytes.
// 252 is evenly divisible by 3 and 4.  Only 192, 10, 241, and 43 are missing.
// Used for ICC profile equivalence testing.
extern const uint8_t skcms_252_random_bytes[252];

// ~~~~ Portable Math ~~~~
static inline float floorf_(float x) {
    float roundtrip = (float)((int)x);
    return roundtrip > x ? roundtrip - 1 : roundtrip;
}
static inline float fabsf_(float x) { return x < 0 ? -x : x; }
float powf_(float, float);

#ifdef __cplusplus
}
#endif
