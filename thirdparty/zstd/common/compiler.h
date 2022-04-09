/*
 * Copyright (c) Yann Collet, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

#ifndef ZSTD_COMPILER_H
#define ZSTD_COMPILER_H

/*-*******************************************************
*  Compiler specifics
*********************************************************/
/* force inlining */

#if !defined(ZSTD_NO_INLINE)
#if (defined(__GNUC__) && !defined(__STRICT_ANSI__)) || defined(__cplusplus) || defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L   /* C99 */
#  define INLINE_KEYWORD inline
#else
#  define INLINE_KEYWORD
#endif

#if defined(__GNUC__) || defined(__ICCARM__)
#  define FORCE_INLINE_ATTR __attribute__((always_inline))
#elif defined(_MSC_VER)
#  define FORCE_INLINE_ATTR __forceinline
#else
#  define FORCE_INLINE_ATTR
#endif

#else

#define INLINE_KEYWORD
#define FORCE_INLINE_ATTR

#endif

/**
  On MSVC qsort requires that functions passed into it use the __cdecl calling conversion(CC).
  This explictly marks such functions as __cdecl so that the code will still compile
  if a CC other than __cdecl has been made the default.
*/
#if  defined(_MSC_VER)
#  define WIN_CDECL __cdecl
#else
#  define WIN_CDECL
#endif

/**
 * FORCE_INLINE_TEMPLATE is used to define C "templates", which take constant
 * parameters. They must be inlined for the compiler to eliminate the constant
 * branches.
 */
#define FORCE_INLINE_TEMPLATE static INLINE_KEYWORD FORCE_INLINE_ATTR
/**
 * HINT_INLINE is used to help the compiler generate better code. It is *not*
 * used for "templates", so it can be tweaked based on the compilers
 * performance.
 *
 * gcc-4.8 and gcc-4.9 have been shown to benefit from leaving off the
 * always_inline attribute.
 *
 * clang up to 5.0.0 (trunk) benefit tremendously from the always_inline
 * attribute.
 */
#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 4 && __GNUC_MINOR__ >= 8 && __GNUC__ < 5
#  define HINT_INLINE static INLINE_KEYWORD
#else
#  define HINT_INLINE static INLINE_KEYWORD FORCE_INLINE_ATTR
#endif

/* UNUSED_ATTR tells the compiler it is okay if the function is unused. */
#if defined(__GNUC__)
#  define UNUSED_ATTR __attribute__((unused))
#else
#  define UNUSED_ATTR
#endif

/* force no inlining */
#ifdef _MSC_VER
#  define FORCE_NOINLINE static __declspec(noinline)
#else
#  if defined(__GNUC__) || defined(__ICCARM__)
#    define FORCE_NOINLINE static __attribute__((__noinline__))
#  else
#    define FORCE_NOINLINE static
#  endif
#endif


/* target attribute */
#ifndef __has_attribute
  #define __has_attribute(x) 0  /* Compatibility with non-clang compilers. */
#endif
#if defined(__GNUC__) || defined(__ICCARM__)
#  define TARGET_ATTRIBUTE(target) __attribute__((__target__(target)))
#else
#  define TARGET_ATTRIBUTE(target)
#endif

/* Enable runtime BMI2 dispatch based on the CPU.
 * Enabled for clang & gcc >=4.8 on x86 when BMI2 isn't enabled by default.
 */
#ifndef DYNAMIC_BMI2
  #if ((defined(__clang__) && __has_attribute(__target__)) \
      || (defined(__GNUC__) \
          && (__GNUC__ >= 5 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8)))) \
      && (defined(__x86_64__) || defined(_M_X86)) \
      && !defined(__BMI2__)
  #  define DYNAMIC_BMI2 1
  #else
  #  define DYNAMIC_BMI2 0
  #endif
#endif

/* prefetch
 * can be disabled, by declaring NO_PREFETCH build macro */
#if defined(NO_PREFETCH)
#  define PREFETCH_L1(ptr)  (void)(ptr)  /* disabled */
#  define PREFETCH_L2(ptr)  (void)(ptr)  /* disabled */
#else
#  if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_I86))  /* _mm_prefetch() is not defined outside of x86/x64 */
#    include <mmintrin.h>   /* https://msdn.microsoft.com/fr-fr/library/84szxsww(v=vs.90).aspx */
#    define PREFETCH_L1(ptr)  _mm_prefetch((const char*)(ptr), _MM_HINT_T0)
#    define PREFETCH_L2(ptr)  _mm_prefetch((const char*)(ptr), _MM_HINT_T1)
#  elif defined(__GNUC__) && ( (__GNUC__ >= 4) || ( (__GNUC__ == 3) && (__GNUC_MINOR__ >= 1) ) )
#    define PREFETCH_L1(ptr)  __builtin_prefetch((ptr), 0 /* rw==read */, 3 /* locality */)
#    define PREFETCH_L2(ptr)  __builtin_prefetch((ptr), 0 /* rw==read */, 2 /* locality */)
#  elif defined(__aarch64__)
#    define PREFETCH_L1(ptr)  __asm__ __volatile__("prfm pldl1keep, %0" ::"Q"(*(ptr)))
#    define PREFETCH_L2(ptr)  __asm__ __volatile__("prfm pldl2keep, %0" ::"Q"(*(ptr)))
#  else
#    define PREFETCH_L1(ptr) (void)(ptr)  /* disabled */
#    define PREFETCH_L2(ptr) (void)(ptr)  /* disabled */
#  endif
#endif  /* NO_PREFETCH */

#define CACHELINE_SIZE 64

#define PREFETCH_AREA(p, s)  {            \
    const char* const _ptr = (const char*)(p);  \
    size_t const _size = (size_t)(s);     \
    size_t _pos;                          \
    for (_pos=0; _pos<_size; _pos+=CACHELINE_SIZE) {  \
        PREFETCH_L2(_ptr + _pos);         \
    }                                     \
}

/* vectorization
 * older GCC (pre gcc-4.3 picked as the cutoff) uses a different syntax */
#if !defined(__INTEL_COMPILER) && !defined(__clang__) && defined(__GNUC__)
#  if (__GNUC__ == 4 && __GNUC_MINOR__ > 3) || (__GNUC__ >= 5)
#    define DONT_VECTORIZE __attribute__((optimize("no-tree-vectorize")))
#  else
#    define DONT_VECTORIZE _Pragma("GCC optimize(\"no-tree-vectorize\")")
#  endif
#else
#  define DONT_VECTORIZE
#endif

/* Tell the compiler that a branch is likely or unlikely.
 * Only use these macros if it causes the compiler to generate better code.
 * If you can remove a LIKELY/UNLIKELY annotation without speed changes in gcc
 * and clang, please do.
 */
#if defined(__GNUC__)
#define LIKELY(x) (__builtin_expect((x), 1))
#define UNLIKELY(x) (__builtin_expect((x), 0))
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

/* disable warnings */
#ifdef _MSC_VER    /* Visual Studio */
#  include <intrin.h>                    /* For Visual 2005 */
#  pragma warning(disable : 4100)        /* disable: C4100: unreferenced formal parameter */
#  pragma warning(disable : 4127)        /* disable: C4127: conditional expression is constant */
#  pragma warning(disable : 4204)        /* disable: C4204: non-constant aggregate initializer */
#  pragma warning(disable : 4214)        /* disable: C4214: non-int bitfields */
#  pragma warning(disable : 4324)        /* disable: C4324: padded structure */
#endif

/*Like DYNAMIC_BMI2 but for compile time determination of BMI2 support*/
#ifndef STATIC_BMI2
#  if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_I86))
#    ifdef __AVX2__  //MSVC does not have a BMI2 specific flag, but every CPU that supports AVX2 also supports BMI2
#       define STATIC_BMI2 1
#    endif
#  endif
#endif

#ifndef STATIC_BMI2
    #define STATIC_BMI2 0
#endif

/* compat. with non-clang compilers */
#ifndef __has_builtin
#  define __has_builtin(x) 0
#endif

/* compat. with non-clang compilers */
#ifndef __has_feature
#  define __has_feature(x) 0
#endif

/* detects whether we are being compiled under msan */
#ifndef ZSTD_MEMORY_SANITIZER
#  if __has_feature(memory_sanitizer)
#    define ZSTD_MEMORY_SANITIZER 1
#  else
#    define ZSTD_MEMORY_SANITIZER 0
#  endif
#endif

#if ZSTD_MEMORY_SANITIZER
/* Not all platforms that support msan provide sanitizers/msan_interface.h.
 * We therefore declare the functions we need ourselves, rather than trying to
 * include the header file... */
#include <stddef.h>  /* size_t */
#define ZSTD_DEPS_NEED_STDINT
#include "zstd_deps.h"  /* intptr_t */

/* Make memory region fully initialized (without changing its contents). */
void __msan_unpoison(const volatile void *a, size_t size);

/* Make memory region fully uninitialized (without changing its contents).
   This is a legacy interface that does not update origin information. Use
   __msan_allocated_memory() instead. */
void __msan_poison(const volatile void *a, size_t size);

/* Returns the offset of the first (at least partially) poisoned byte in the
   memory range, or -1 if the whole range is good. */
intptr_t __msan_test_shadow(const volatile void *x, size_t size);
#endif

/* detects whether we are being compiled under asan */
#ifndef ZSTD_ADDRESS_SANITIZER
#  if __has_feature(address_sanitizer)
#    define ZSTD_ADDRESS_SANITIZER 1
#  elif defined(__SANITIZE_ADDRESS__)
#    define ZSTD_ADDRESS_SANITIZER 1
#  else
#    define ZSTD_ADDRESS_SANITIZER 0
#  endif
#endif

#if ZSTD_ADDRESS_SANITIZER
/* Not all platforms that support asan provide sanitizers/asan_interface.h.
 * We therefore declare the functions we need ourselves, rather than trying to
 * include the header file... */
#include <stddef.h>  /* size_t */

/**
 * Marks a memory region (<c>[addr, addr+size)</c>) as unaddressable.
 *
 * This memory must be previously allocated by your program. Instrumented
 * code is forbidden from accessing addresses in this region until it is
 * unpoisoned. This function is not guaranteed to poison the entire region -
 * it could poison only a subregion of <c>[addr, addr+size)</c> due to ASan
 * alignment restrictions.
 *
 * \note This function is not thread-safe because no two threads can poison or
 * unpoison memory in the same memory region simultaneously.
 *
 * \param addr Start of memory region.
 * \param size Size of memory region. */
void __asan_poison_memory_region(void const volatile *addr, size_t size);

/**
 * Marks a memory region (<c>[addr, addr+size)</c>) as addressable.
 *
 * This memory must be previously allocated by your program. Accessing
 * addresses in this region is allowed until this region is poisoned again.
 * This function could unpoison a super-region of <c>[addr, addr+size)</c> due
 * to ASan alignment restrictions.
 *
 * \note This function is not thread-safe because no two threads can
 * poison or unpoison memory in the same memory region simultaneously.
 *
 * \param addr Start of memory region.
 * \param size Size of memory region. */
void __asan_unpoison_memory_region(void const volatile *addr, size_t size);
#endif

#endif /* ZSTD_COMPILER_H */
