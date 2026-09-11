// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_BASE_COMPILER_SPECIFIC_H_
#define LIB_JXL_BASE_COMPILER_SPECIFIC_H_

// Macros for compiler version + nonstandard keywords, e.g. __builtin_expect.

#include <sys/types.h>

#include "lib/jxl/base/sanitizer_definitions.h"

#if JXL_ADDRESS_SANITIZER || JXL_MEMORY_SANITIZER || JXL_THREAD_SANITIZER
#include "sanitizer/common_interface_defs.h"  // __sanitizer_print_stack_trace
#endif                                        // defined(*_SANITIZER)

// #if is shorter and safer than #ifdef. *_VERSION are zero if not detected,
// otherwise 100 * major + minor version. Note that other packages check for
// #ifdef COMPILER_MSVC, so we cannot use that same name.

#ifdef _MSC_VER
#define JXL_COMPILER_MSVC _MSC_VER
#else
#define JXL_COMPILER_MSVC 0
#endif

#ifdef __GNUC__
#define JXL_COMPILER_GCC (__GNUC__ * 100 + __GNUC_MINOR__)
#else
#define JXL_COMPILER_GCC 0
#endif

#ifdef __clang__
#define JXL_COMPILER_CLANG (__clang_major__ * 100 + __clang_minor__)
// Clang pretends to be GCC for compatibility.
#undef JXL_COMPILER_GCC
#define JXL_COMPILER_GCC 0
#else
#define JXL_COMPILER_CLANG 0
#endif

#if JXL_COMPILER_MSVC
#define JXL_RESTRICT __restrict
#elif JXL_COMPILER_GCC || JXL_COMPILER_CLANG
#define JXL_RESTRICT __restrict__
#else
#define JXL_RESTRICT
#endif

#if JXL_COMPILER_MSVC
#define JXL_INLINE __forceinline
#define JXL_NOINLINE __declspec(noinline)
#else
#define JXL_INLINE inline __attribute__((always_inline))
#define JXL_NOINLINE __attribute__((noinline))
#endif

#if JXL_COMPILER_MSVC
#define JXL_NORETURN __declspec(noreturn)
#elif JXL_COMPILER_GCC || JXL_COMPILER_CLANG
#define JXL_NORETURN __attribute__((noreturn))
#else
#define JXL_NORETURN
#endif

#if JXL_COMPILER_MSVC
#define JXL_MAYBE_UNUSED
#else
// Encountered "attribute list cannot appear here" when using the C++17
// [[maybe_unused]], so only use the old style attribute for now.
#define JXL_MAYBE_UNUSED __attribute__((unused))
#endif

// MSAN execution won't hurt if some code it not inlined, but this can greatly
// improve compilation time. Unfortunately this macro can not be used just
// everywhere - inside header files it leads to "multiple definition" error;
// though it would be better not to have JXL_INLINE in header overall.
#if JXL_MEMORY_SANITIZER || JXL_ADDRESS_SANITIZER || JXL_THREAD_SANITIZER
#define JXL_MAYBE_INLINE JXL_MAYBE_UNUSED
#else
#define JXL_MAYBE_INLINE JXL_INLINE
#endif

#if JXL_COMPILER_MSVC
// Unsupported, __assume is not the same.
#define JXL_LIKELY(expr) expr
#define JXL_UNLIKELY(expr) expr
#else
#define JXL_LIKELY(expr) __builtin_expect(!!(expr), 1)
#define JXL_UNLIKELY(expr) __builtin_expect(!!(expr), 0)
#endif

#if JXL_COMPILER_MSVC
#include <stdint.h>
using ssize_t = intptr_t;
#endif

// Returns a void* pointer which the compiler then assumes is N-byte aligned.
// Example: float* JXL_RESTRICT aligned = (float*)JXL_ASSUME_ALIGNED(in, 32);
//
// The assignment semantics are required by GCC/Clang. ICC provides an in-place
// __assume_aligned, whereas MSVC's __assume appears unsuitable.
#if JXL_COMPILER_CLANG
// Early versions of Clang did not support __builtin_assume_aligned.
#define JXL_HAS_ASSUME_ALIGNED __has_builtin(__builtin_assume_aligned)
#elif JXL_COMPILER_GCC
#define JXL_HAS_ASSUME_ALIGNED 1
#else
#define JXL_HAS_ASSUME_ALIGNED 0
#endif

#if JXL_HAS_ASSUME_ALIGNED
#define JXL_ASSUME_ALIGNED(ptr, align) __builtin_assume_aligned((ptr), (align))
#else
#define JXL_ASSUME_ALIGNED(ptr, align) (ptr) /* not supported */
#endif

#ifdef __has_attribute
#define JXL_HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define JXL_HAVE_ATTRIBUTE(x) 0
#endif

// Raises warnings if the function return value is unused. Should appear as the
// first part of a function definition/declaration.
#if JXL_HAVE_ATTRIBUTE(nodiscard)
#define JXL_MUST_USE_RESULT [[nodiscard]]
#elif JXL_COMPILER_CLANG && JXL_HAVE_ATTRIBUTE(warn_unused_result)
#define JXL_MUST_USE_RESULT __attribute__((warn_unused_result))
#else
#define JXL_MUST_USE_RESULT
#endif

// Disable certain -fsanitize flags for functions that are expected to include
// things like unsigned integer overflow. For example use in the function
// declaration JXL_NO_SANITIZE("unsigned-integer-overflow") to silence unsigned
// integer overflow ubsan messages.
#if JXL_COMPILER_CLANG && JXL_HAVE_ATTRIBUTE(no_sanitize)
#define JXL_NO_SANITIZE(X) __attribute__((no_sanitize(X)))
#else
#define JXL_NO_SANITIZE(X)
#endif

#if JXL_HAVE_ATTRIBUTE(__format__)
#define JXL_FORMAT(idx_fmt, idx_arg) \
  __attribute__((__format__(__printf__, idx_fmt, idx_arg)))
#else
#define JXL_FORMAT(idx_fmt, idx_arg)
#endif

// C++ standard.
#if defined(_MSC_VER) && !defined(__clang__) && defined(_MSVC_LANG) && \
    _MSVC_LANG > __cplusplus
#define JXL_CXX_LANG _MSVC_LANG
#else
#define JXL_CXX_LANG __cplusplus
#endif

// Known / distinguished C++ standards.
#define JXL_CXX_17 201703

// In most cases we consider build as "debug". Use `NDEBUG` for release build.
#if defined(JXL_IS_DEBUG_BUILD)
#undef JXL_IS_DEBUG_BUILD
#define JXL_IS_DEBUG_BUILD 1
#elif defined(NDEBUG)
#define JXL_IS_DEBUG_BUILD 0
#else
#define JXL_IS_DEBUG_BUILD 1
#endif

#if defined(JXL_CRASH_ON_ERROR)
#undef JXL_CRASH_ON_ERROR
#define JXL_CRASH_ON_ERROR 1
#else
#define JXL_CRASH_ON_ERROR 0
#endif

#if JXL_CRASH_ON_ERROR && !JXL_IS_DEBUG_BUILD
#error "JXL_CRASH_ON_ERROR requires JXL_IS_DEBUG_BUILD"
#endif

// Pass -DJXL_DEBUG_ON_ALL_ERROR at compile time to print debug messages on
// all error (fatal and non-fatal) status.
#if defined(JXL_DEBUG_ON_ALL_ERROR)
#undef JXL_DEBUG_ON_ALL_ERROR
#define JXL_DEBUG_ON_ALL_ERROR 1
#else
#define JXL_DEBUG_ON_ALL_ERROR 0
#endif

#if JXL_DEBUG_ON_ALL_ERROR && !JXL_IS_DEBUG_BUILD
#error "JXL_DEBUG_ON_ALL_ERROR requires JXL_IS_DEBUG_BUILD"
#endif

// Pass -DJXL_DEBUG_ON_ABORT={0} to disable the debug messages on
// (debug) JXL_ENSURE and JXL_DASSERT.
#if !defined(JXL_DEBUG_ON_ABORT)
#define JXL_DEBUG_ON_ABORT JXL_IS_DEBUG_BUILD
#endif  // JXL_DEBUG_ON_ABORT

#if JXL_DEBUG_ON_ABORT && !JXL_IS_DEBUG_BUILD
#error "JXL_DEBUG_ON_ABORT requires JXL_IS_DEBUG_BUILD"
#endif

#if JXL_ADDRESS_SANITIZER || JXL_MEMORY_SANITIZER || JXL_THREAD_SANITIZER
#define JXL_PRINT_STACK_TRACE() __sanitizer_print_stack_trace();
#else
#define JXL_PRINT_STACK_TRACE()
#endif

#if JXL_COMPILER_MSVC
#define JXL_CRASH() __debugbreak(), (void)abort()
#else
#define JXL_CRASH() (void)__builtin_trap()
#endif

#endif  // LIB_JXL_BASE_COMPILER_SPECIFIC_H_
