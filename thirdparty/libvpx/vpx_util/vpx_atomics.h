/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_UTIL_VPX_ATOMICS_H_
#define VPX_VPX_UTIL_VPX_ATOMICS_H_

#include "./vpx_config.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#if CONFIG_OS_SUPPORT && CONFIG_MULTITHREAD

// Look for built-in atomic support. We cannot use <stdatomic.h> or <atomic>
// since neither is guaranteed to exist on both C and C++ platforms, and we need
// to back the atomic type with the same type (g++ needs to be able to use
// gcc-built code). g++ 6 doesn't support _Atomic as a keyword and can't use the
// stdatomic.h header. Even if both <stdatomic.h> and <atomic> existed it's not
// guaranteed that atomic_int is the same type as std::atomic_int.
// See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=60932#c13.
#if !defined(__has_builtin)
#define __has_builtin(x) 0  // Compatibility with non-clang compilers.
#endif                      // !defined(__has_builtin)

#if (__has_builtin(__atomic_load_n)) || \
    (defined(__GNUC__) &&               \
     (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 7)))
// For GCC >= 4.7 and Clang versions that support __atomic builtins, use those.
#define VPX_USE_ATOMIC_BUILTINS
#else
// Use platform-specific asm barriers.
#if defined(_MSC_VER)
// TODO(pbos): This assumes that newer versions of MSVC are building with the
// default /volatile:ms (or older, where this is always true. Consider adding
// support for using <atomic> instead of stdatomic.h when building C++11 under
// MSVC. It's unclear what to do for plain C under /volatile:iso (inline asm?),
// there're no explicit Interlocked* functions for only storing or loading
// (presumably because volatile has historically implied that on MSVC).
//
// For earlier versions of MSVC or the default /volatile:ms volatile int are
// acquire/release and require no barrier.
#define vpx_atomic_memory_barrier() \
  do {                              \
  } while (0)
#else
#if VPX_ARCH_X86 || VPX_ARCH_X86_64
// Use a compiler barrier on x86, no runtime penalty.
#define vpx_atomic_memory_barrier() __asm__ __volatile__("" ::: "memory")
#elif VPX_ARCH_ARM
#define vpx_atomic_memory_barrier() __asm__ __volatile__("dmb ish" ::: "memory")
#elif VPX_ARCH_MIPS
#define vpx_atomic_memory_barrier() __asm__ __volatile__("sync" ::: "memory")
#else
#error Unsupported architecture!
#endif  // VPX_ARCH_X86 || VPX_ARCH_X86_64
#endif  // defined(_MSC_VER)
#endif  // atomic builtin availability check

// These are wrapped in a struct so that they are not easily accessed directly
// on any platform (to discourage programmer errors by setting values directly).
// This primitive MUST be initialized using vpx_atomic_init or VPX_ATOMIC_INIT
// (NOT memset) and accessed through vpx_atomic_ functions.
typedef struct vpx_atomic_int {
  volatile int value;
} vpx_atomic_int;

#define VPX_ATOMIC_INIT(num) { num }

// Initialization of an atomic int, not thread safe.
static INLINE void vpx_atomic_init(vpx_atomic_int *atomic, int value) {
  atomic->value = value;
}

static INLINE void vpx_atomic_store_release(vpx_atomic_int *atomic, int value) {
#if defined(VPX_USE_ATOMIC_BUILTINS)
  __atomic_store_n(&atomic->value, value, __ATOMIC_RELEASE);
#else
  vpx_atomic_memory_barrier();
  atomic->value = value;
#endif  // defined(VPX_USE_ATOMIC_BUILTINS)
}

static INLINE int vpx_atomic_load_acquire(const vpx_atomic_int *atomic) {
#if defined(VPX_USE_ATOMIC_BUILTINS)
  return __atomic_load_n(&atomic->value, __ATOMIC_ACQUIRE);
#else
  int v = atomic->value;
  vpx_atomic_memory_barrier();
  return v;
#endif  // defined(VPX_USE_ATOMIC_BUILTINS)
}

#undef VPX_USE_ATOMIC_BUILTINS
#undef vpx_atomic_memory_barrier

#endif /* CONFIG_OS_SUPPORT && CONFIG_MULTITHREAD */

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // VPX_VPX_UTIL_VPX_ATOMICS_H_
