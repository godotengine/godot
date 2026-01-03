/*
 *  Copyright 2024 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef INCLUDE_LIBYUV_CPU_SUPPORT_H_
#define INCLUDE_LIBYUV_CPU_SUPPORT_H_

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

#if defined(__pnacl__) || defined(__CLR_VER) ||            \
    (defined(__native_client__) && defined(__x86_64__)) || \
    (defined(__i386__) && !defined(__SSE__) && !defined(__clang__))
#define LIBYUV_DISABLE_X86
#endif

#if defined(__native_client__)
#define LIBYUV_DISABLE_NEON
#endif

// MemorySanitizer does not support assembly code yet. http://crbug.com/344505
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#if !defined(LIBYUV_DISABLE_NEON)
#define LIBYUV_DISABLE_NEON
#endif
#if !defined(LIBYUV_DISABLE_SME)
#define LIBYUV_DISABLE_SME
#endif
#if !defined(LIBYUV_DISABLE_SVE)
#define LIBYUV_DISABLE_SVE
#endif
#if !defined(LIBYUV_DISABLE_X86)
#define LIBYUV_DISABLE_X86
#endif
#endif  // __has_feature(memory_sanitizer)
#endif  // defined(__has_feature)

// clang >= 3.5.0 required for Arm64.
#if defined(__clang__) && defined(__aarch64__) && !defined(LIBYUV_DISABLE_NEON)
#if (__clang_major__ < 3) || (__clang_major__ == 3 && (__clang_minor__ < 5))
#define LIBYUV_DISABLE_NEON
#endif  // clang >= 3.5
#endif  // __clang__

// GCC >= 4.7.0 required for AVX2.
#if defined(__GNUC__) && !defined(LIBYUV_ENABLE_ROWWIN) && \
    (defined(__x86_64__) || defined(__i386__))
#if (__GNUC__ > 4) || (__GNUC__ == 4 && (__GNUC_MINOR__ >= 7))
#define GCC_HAS_AVX2 1
#endif  // GNUC >= 4.7
#endif  // __GNUC__

// clang >= 3.4.0 required for AVX2.
#if defined(__clang__) && !defined(LIBYUV_ENABLE_ROWWIN) && \
    (defined(__x86_64__) || defined(__i386__))
#if (__clang_major__ > 3) || (__clang_major__ == 3 && (__clang_minor__ >= 4))
#define CLANG_HAS_AVX2 1
#endif  // clang >= 3.4
#endif  // __clang__

// clang >= 6.0.0 required for AVX512.
#if defined(__clang__) && !defined(LIBYUV_ENABLE_ROWWIN) && \
    (defined(__x86_64__) || defined(__i386__))
// clang in xcode follows a different versioning scheme.
// TODO(fbarchard): fix xcode 9 ios b/789.
#if (__clang_major__ >= 7) && !defined(__APPLE__)
#define CLANG_HAS_AVX512 1
#endif  // clang >= 7
#endif  // __clang__

// Visual C 2012 required for AVX2.
#if defined(_M_IX86) &&                                       \
    (!defined(__clang__) || defined(LIBYUV_ENABLE_ROWWIN)) && \
    defined(_MSC_VER) && _MSC_VER >= 1700
#define VISUALC_HAS_AVX2 1
#endif  // VisualStudio >= 2012

// Clang 19 required for SME due to needing __arm_tpidr2_save from compiler-rt,
// only enabled on Linux and Android (both define __linux__) for now.
#if !defined(LIBYUV_DISABLE_SME) && defined(__aarch64__) && \
    defined(__linux__) && defined(__clang__) && (__clang_major__ >= 19)
#define CLANG_HAS_SME 1
#endif

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // INCLUDE_LIBYUV_CPU_SUPPORT_H_
