// Copyright 2022 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
//   CPU detection functions and macros.
//
// Author: Skal (pascal.massimino@gmail.com)

#ifndef WEBP_DSP_CPU_H_
#define WEBP_DSP_CPU_H_

#ifdef HAVE_CONFIG_H
#include "src/webp/config.h"
#endif

#include "src/webp/types.h"

#if defined(__GNUC__)
#define LOCAL_GCC_VERSION ((__GNUC__ << 8) | __GNUC_MINOR__)
#define LOCAL_GCC_PREREQ(maj, min) (LOCAL_GCC_VERSION >= (((maj) << 8) | (min)))
#else
#define LOCAL_GCC_VERSION 0
#define LOCAL_GCC_PREREQ(maj, min) 0
#endif

#if defined(__clang__)
#define LOCAL_CLANG_VERSION ((__clang_major__ << 8) | __clang_minor__)
#define LOCAL_CLANG_PREREQ(maj, min) \
  (LOCAL_CLANG_VERSION >= (((maj) << 8) | (min)))
#else
#define LOCAL_CLANG_VERSION 0
#define LOCAL_CLANG_PREREQ(maj, min) 0
#endif

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#if !defined(HAVE_CONFIG_H)
#if defined(_MSC_VER) && _MSC_VER > 1310 && \
    (defined(_M_X64) || defined(_M_IX86))
#define WEBP_MSC_SSE2  // Visual C++ SSE2 targets
#endif

#if defined(_MSC_VER) && _MSC_VER >= 1500 && \
    (defined(_M_X64) || defined(_M_IX86))
#define WEBP_MSC_SSE41  // Visual C++ SSE4.1 targets
#endif
#endif

// WEBP_HAVE_* are used to indicate the presence of the instruction set in dsp
// files without intrinsics, allowing the corresponding Init() to be called.
// Files containing intrinsics will need to be built targeting the instruction
// set so should succeed on one of the earlier tests.
#if (defined(__SSE2__) || defined(WEBP_MSC_SSE2)) && \
    (!defined(HAVE_CONFIG_H) || defined(WEBP_HAVE_SSE2))
#define WEBP_USE_SSE2
#endif

#if defined(WEBP_USE_SSE2) && !defined(WEBP_HAVE_SSE2)
#define WEBP_HAVE_SSE2
#endif

#if (defined(__SSE4_1__) || defined(WEBP_MSC_SSE41)) && \
    (!defined(HAVE_CONFIG_H) || defined(WEBP_HAVE_SSE41))
#define WEBP_USE_SSE41
#endif

#if defined(WEBP_USE_SSE41) && !defined(WEBP_HAVE_SSE41)
#define WEBP_HAVE_SSE41
#endif

#undef WEBP_MSC_SSE41
#undef WEBP_MSC_SSE2

// The intrinsics currently cause compiler errors with arm-nacl-gcc and the
// inline assembly would need to be modified for use with Native Client.
#if ((defined(__ARM_NEON__) || defined(__aarch64__)) &&       \
     (!defined(HAVE_CONFIG_H) || defined(WEBP_HAVE_NEON))) && \
    !defined(__native_client__)
#define WEBP_USE_NEON
#endif

#if !defined(WEBP_USE_NEON) && defined(__ANDROID__) && \
    defined(__ARM_ARCH_7A__) && defined(HAVE_CPU_FEATURES_H)
#define WEBP_ANDROID_NEON  // Android targets that may have NEON
#define WEBP_USE_NEON
#endif

// Note: ARM64 is supported in Visual Studio 2017, but requires the direct
// inclusion of arm64_neon.h; Visual Studio 2019 includes this file in
// arm_neon.h. Compile errors were seen with Visual Studio 2019 16.4 with
// vtbl4_u8(); a fix was made in 16.6.
#if defined(_MSC_VER) && ((_MSC_VER >= 1700 && defined(_M_ARM)) || \
                          (_MSC_VER >= 1926 && defined(_M_ARM64)))
#define WEBP_USE_NEON
#define WEBP_USE_INTRINSICS
#endif

#if defined(WEBP_USE_NEON) && !defined(WEBP_HAVE_NEON)
#define WEBP_HAVE_NEON
#endif

#if defined(__mips__) && !defined(__mips64) && defined(__mips_isa_rev) && \
    (__mips_isa_rev >= 1) && (__mips_isa_rev < 6)
#define WEBP_USE_MIPS32
#if (__mips_isa_rev >= 2)
#define WEBP_USE_MIPS32_R2
#if defined(__mips_dspr2) || (defined(__mips_dsp_rev) && __mips_dsp_rev >= 2)
#define WEBP_USE_MIPS_DSP_R2
#endif
#endif
#endif

#if defined(__mips_msa) && defined(__mips_isa_rev) && (__mips_isa_rev >= 5)
#define WEBP_USE_MSA
#endif

#ifndef WEBP_DSP_OMIT_C_CODE
#define WEBP_DSP_OMIT_C_CODE 1
#endif

#if defined(WEBP_USE_NEON) && WEBP_DSP_OMIT_C_CODE
#define WEBP_NEON_OMIT_C_CODE 1
#else
#define WEBP_NEON_OMIT_C_CODE 0
#endif

#if !(LOCAL_CLANG_PREREQ(3, 8) || LOCAL_GCC_PREREQ(4, 8) || \
      defined(__aarch64__))
#define WEBP_NEON_WORK_AROUND_GCC 1
#else
#define WEBP_NEON_WORK_AROUND_GCC 0
#endif

// This macro prevents thread_sanitizer from reporting known concurrent writes.
#define WEBP_TSAN_IGNORE_FUNCTION
#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#undef WEBP_TSAN_IGNORE_FUNCTION
#define WEBP_TSAN_IGNORE_FUNCTION __attribute__((no_sanitize_thread))
#endif
#endif

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#define WEBP_MSAN
#endif
#endif

#if defined(WEBP_USE_THREAD) && !defined(_WIN32)
#include <pthread.h>  // NOLINT

#define WEBP_DSP_INIT(func)                                         \
  do {                                                              \
    static volatile VP8CPUInfo func##_last_cpuinfo_used =           \
        (VP8CPUInfo)&func##_last_cpuinfo_used;                      \
    static pthread_mutex_t func##_lock = PTHREAD_MUTEX_INITIALIZER; \
    if (pthread_mutex_lock(&func##_lock)) break;                    \
    if (func##_last_cpuinfo_used != VP8GetCPUInfo) func();          \
    func##_last_cpuinfo_used = VP8GetCPUInfo;                       \
    (void)pthread_mutex_unlock(&func##_lock);                       \
  } while (0)
#else  // !(defined(WEBP_USE_THREAD) && !defined(_WIN32))
#define WEBP_DSP_INIT(func)                               \
  do {                                                    \
    static volatile VP8CPUInfo func##_last_cpuinfo_used = \
        (VP8CPUInfo)&func##_last_cpuinfo_used;            \
    if (func##_last_cpuinfo_used == VP8GetCPUInfo) break; \
    func();                                               \
    func##_last_cpuinfo_used = VP8GetCPUInfo;             \
  } while (0)
#endif  // defined(WEBP_USE_THREAD) && !defined(_WIN32)

// Defines an Init + helper function that control multiple initialization of
// function pointers / tables.
/* Usage:
   WEBP_DSP_INIT_FUNC(InitFunc) {
     ...function body
   }
*/
#define WEBP_DSP_INIT_FUNC(name)                                            \
  static WEBP_TSAN_IGNORE_FUNCTION void name##_body(void);                  \
  WEBP_TSAN_IGNORE_FUNCTION void name(void) { WEBP_DSP_INIT(name##_body); } \
  static WEBP_TSAN_IGNORE_FUNCTION void name##_body(void)

#define WEBP_UBSAN_IGNORE_UNDEF
#define WEBP_UBSAN_IGNORE_UNSIGNED_OVERFLOW
#if defined(__clang__) && defined(__has_attribute)
#if __has_attribute(no_sanitize)
// This macro prevents the undefined behavior sanitizer from reporting
// failures. This is only meant to silence unaligned loads on platforms that
// are known to support them.
#undef WEBP_UBSAN_IGNORE_UNDEF
#define WEBP_UBSAN_IGNORE_UNDEF __attribute__((no_sanitize("undefined")))

// This macro prevents the undefined behavior sanitizer from reporting
// failures related to unsigned integer overflows. This is only meant to
// silence cases where this well defined behavior is expected.
#undef WEBP_UBSAN_IGNORE_UNSIGNED_OVERFLOW
#define WEBP_UBSAN_IGNORE_UNSIGNED_OVERFLOW \
  __attribute__((no_sanitize("unsigned-integer-overflow")))
#endif
#endif

// If 'ptr' is NULL, returns NULL. Otherwise returns 'ptr + off'.
// Prevents undefined behavior sanitizer nullptr-with-nonzero-offset warning.
#if !defined(WEBP_OFFSET_PTR)
#define WEBP_OFFSET_PTR(ptr, off) (((ptr) == NULL) ? NULL : ((ptr) + (off)))
#endif

// Regularize the definition of WEBP_SWAP_16BIT_CSP (backward compatibility)
#if !defined(WEBP_SWAP_16BIT_CSP)
#define WEBP_SWAP_16BIT_CSP 0
#endif

// some endian fix (e.g.: mips-gcc doesn't define __BIG_ENDIAN__)
#if !defined(WORDS_BIGENDIAN) &&                   \
    (defined(__BIG_ENDIAN__) || defined(_M_PPC) || \
     (defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)))
#define WORDS_BIGENDIAN
#endif

typedef enum {
  kSSE2,
  kSSE3,
  kSlowSSSE3,  // special feature for slow SSSE3 architectures
  kSSE4_1,
  kAVX,
  kAVX2,
  kNEON,
  kMIPS32,
  kMIPSdspR2,
  kMSA
} CPUFeature;

#ifdef __cplusplus
extern "C" {
#endif

// returns true if the CPU supports the feature.
typedef int (*VP8CPUInfo)(CPUFeature feature);
WEBP_EXTERN VP8CPUInfo VP8GetCPUInfo;

#ifdef __cplusplus
}    // extern "C"
#endif

#endif  // WEBP_DSP_CPU_H_
