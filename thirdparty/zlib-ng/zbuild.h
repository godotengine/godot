#ifndef _ZBUILD_H
#define _ZBUILD_H

#define _POSIX_SOURCE 1  /* fileno */
#ifndef _POSIX_C_SOURCE
#  define _POSIX_C_SOURCE 200809L /* snprintf, posix_memalign, strdup */
#endif
#ifndef _ISOC11_SOURCE
#  define _ISOC11_SOURCE 1 /* aligned_alloc */
#endif
#ifdef __OpenBSD__
#  define _BSD_SOURCE 1
#endif

#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

/* Determine compiler version of C Standard */
#ifdef __STDC_VERSION__
#  if __STDC_VERSION__ >= 199901L
#    ifndef STDC99
#      define STDC99
#    endif
#  endif
#  if __STDC_VERSION__ >= 201112L
#    ifndef STDC11
#      define STDC11
#    endif
#  endif
#endif

#ifndef Z_HAS_ATTRIBUTE
#  if defined(__has_attribute)
#    define Z_HAS_ATTRIBUTE(a) __has_attribute(a)
#  else
#    define Z_HAS_ATTRIBUTE(a) 0
#  endif
#endif

#ifndef Z_FALLTHROUGH
#  if Z_HAS_ATTRIBUTE(__fallthrough__) || (defined(__GNUC__) && (__GNUC__ >= 7))
#    define Z_FALLTHROUGH __attribute__((__fallthrough__))
#  else
#    define Z_FALLTHROUGH do {} while(0) /* fallthrough */
#  endif
#endif

#ifndef Z_TARGET
#  if Z_HAS_ATTRIBUTE(__target__)
#    define Z_TARGET(x) __attribute__((__target__(x)))
#  else
#    define Z_TARGET(x)
#  endif
#endif

/* This has to be first include that defines any types */
#if defined(_MSC_VER)
#  if defined(_WIN64)
    typedef __int64 ssize_t;
#  else
    typedef long ssize_t;
#  endif

#  if defined(_WIN64)
    #define SSIZE_MAX _I64_MAX
#  else
    #define SSIZE_MAX LONG_MAX
#  endif
#endif

/* A forced inline decorator */
#if defined(_MSC_VER)
#  define Z_FORCEINLINE __forceinline
#elif defined(__GNUC__)
#  define Z_FORCEINLINE inline __attribute__((always_inline))
#else
    /* It won't actually force inlining but it will suggest it */
#  define Z_FORCEINLINE inline
#endif

/* MS Visual Studio does not allow inline in C, only C++.
   But it provides __inline instead, so use that. */
#if defined(_MSC_VER) && !defined(inline) && !defined(__cplusplus)
#  define inline __inline
#endif

#if defined(ZLIB_COMPAT)
#  define PREFIX(x) x
#  define PREFIX2(x) ZLIB_ ## x
#  define PREFIX3(x) z_ ## x
#  define PREFIX4(x) x ## 64
#  define zVersion zlibVersion
#else
#  define PREFIX(x) zng_ ## x
#  define PREFIX2(x) ZLIBNG_ ## x
#  define PREFIX3(x) zng_ ## x
#  define PREFIX4(x) zng_ ## x
#  define zVersion zlibng_version
#  define z_size_t size_t
#endif

/* In zlib-compat some functions and types use unsigned long, but zlib-ng use size_t */
#if defined(ZLIB_COMPAT)
#  define z_uintmax_t unsigned long
#else
#  define z_uintmax_t size_t
#endif

/* In zlib-compat headers some function return values and parameter types use int or unsigned, but zlib-ng headers use
   int32_t and uint32_t, which will cause type mismatch when compiling zlib-ng if int32_t is long and uint32_t is
   unsigned long */
#if defined(ZLIB_COMPAT)
#  define z_int32_t int
#  define z_uint32_t unsigned int
#else
#  define z_int32_t int32_t
#  define z_uint32_t uint32_t
#endif

/* Minimum of a and b. */
#define MIN(a, b) ((a) > (b) ? (b) : (a))
/* Maximum of a and b. */
#define MAX(a, b) ((a) < (b) ? (b) : (a))
/* Ignore unused variable warning */
#define Z_UNUSED(var) (void)(var)

#if defined(HAVE_VISIBILITY_INTERNAL)
#  define Z_INTERNAL __attribute__((visibility ("internal")))
#elif defined(HAVE_VISIBILITY_HIDDEN)
#  define Z_INTERNAL __attribute__((visibility ("hidden")))
#else
#  define Z_INTERNAL
#endif

/* Symbol versioning helpers, allowing multiple versions of a function to exist.
 * Functions using this must also be added to zlib-ng.map for each version.
 * Double @@ means this is the default for newly compiled applications to link against.
 * Single @ means this is kept for backwards compatibility.
 * This is only used for Zlib-ng native API, and only on platforms supporting this.
 */
#if defined(HAVE_SYMVER)
#  define ZSYMVER(func,alias,ver) __asm__(".symver " func ", " alias "@ZLIB_NG_" ver);
#  define ZSYMVER_DEF(func,alias,ver) __asm__(".symver " func ", " alias "@@ZLIB_NG_" ver);
#else
#  define ZSYMVER(func,alias,ver)
#  define ZSYMVER_DEF(func,alias,ver)
#endif

#ifndef __cplusplus
#  define Z_REGISTER register
#else
#  define Z_REGISTER
#endif

/* Reverse the bytes in a value. Use compiler intrinsics when
   possible to take advantage of hardware implementations. */
#if defined(_MSC_VER) && (_MSC_VER >= 1300)
#  include <stdlib.h>
#  pragma intrinsic(_byteswap_ulong)
#  define ZSWAP16(q) _byteswap_ushort(q)
#  define ZSWAP32(q) _byteswap_ulong(q)
#  define ZSWAP64(q) _byteswap_uint64(q)

#elif defined(__clang__) || (defined(__GNUC__) && \
        (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8)))
#  define ZSWAP16(q) __builtin_bswap16(q)
#  define ZSWAP32(q) __builtin_bswap32(q)
#  define ZSWAP64(q) __builtin_bswap64(q)

#elif defined(__GNUC__) && (__GNUC__ >= 2) && defined(__linux__)
#  include <byteswap.h>
#  define ZSWAP16(q) bswap_16(q)
#  define ZSWAP32(q) bswap_32(q)
#  define ZSWAP64(q) bswap_64(q)

#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__DragonFly__)
#  include <sys/endian.h>
#  define ZSWAP16(q) bswap16(q)
#  define ZSWAP32(q) bswap32(q)
#  define ZSWAP64(q) bswap64(q)
#elif defined(__OpenBSD__)
#  include <sys/endian.h>
#  define ZSWAP16(q) swap16(q)
#  define ZSWAP32(q) swap32(q)
#  define ZSWAP64(q) swap64(q)
#elif defined(__INTEL_COMPILER)
/* ICC does not provide a two byte swap. */
#  define ZSWAP16(q) ((((q) & 0xff) << 8) | (((q) & 0xff00) >> 8))
#  define ZSWAP32(q) _bswap(q)
#  define ZSWAP64(q) _bswap64(q)

#else
#  define ZSWAP16(q) ((((q) & 0xff) << 8) | (((q) & 0xff00) >> 8))
#  define ZSWAP32(q) ((((q) >> 24) & 0xff) + (((q) >> 8) & 0xff00) + \
                     (((q) & 0xff00) << 8) + (((q) & 0xff) << 24))
#  define ZSWAP64(q)                           \
         (((q & 0xFF00000000000000u) >> 56u) | \
          ((q & 0x00FF000000000000u) >> 40u) | \
          ((q & 0x0000FF0000000000u) >> 24u) | \
          ((q & 0x000000FF00000000u) >> 8u)  | \
          ((q & 0x00000000FF000000u) << 8u)  | \
          ((q & 0x0000000000FF0000u) << 24u) | \
          ((q & 0x000000000000FF00u) << 40u) | \
          ((q & 0x00000000000000FFu) << 56u))
#endif

/* Only enable likely/unlikely if the compiler is known to support it */
#if (defined(__GNUC__) && (__GNUC__ >= 3)) || defined(__INTEL_COMPILER) || defined(__clang__)
#  define LIKELY_NULL(x)        __builtin_expect((x) != 0, 0)
#  define LIKELY(x)             __builtin_expect(!!(x), 1)
#  define UNLIKELY(x)           __builtin_expect(!!(x), 0)
#else
#  define LIKELY_NULL(x)        x
#  define LIKELY(x)             x
#  define UNLIKELY(x)           x
#endif /* (un)likely */

#if defined(HAVE_ATTRIBUTE_ALIGNED)
#  define ALIGNED_(x) __attribute__ ((aligned(x)))
#elif defined(_MSC_VER)
#  define ALIGNED_(x) __declspec(align(x))
#else
/* TODO: Define ALIGNED_ for your compiler */
#  define ALIGNED_(x)
#endif

#ifdef HAVE_BUILTIN_ASSUME_ALIGNED
#  define HINT_ALIGNED(p,n) __builtin_assume_aligned((void *)(p),(n))
#else
#  define HINT_ALIGNED(p,n) (p)
#endif
#define HINT_ALIGNED_16(p) HINT_ALIGNED((p),16)
#define HINT_ALIGNED_64(p) HINT_ALIGNED((p),64)
#define HINT_ALIGNED_4096(p) HINT_ALIGNED((p),4096)

/* PADSZ returns needed bytes to pad bpos to pad size
 * PAD_NN calculates pad size and adds it to bpos, returning the result.
 * All take an integer or a pointer as bpos input.
 */
#define PADSZ(bpos, pad) (((pad) - ((uintptr_t)(bpos) % (pad))) % (pad))
#define PAD_16(bpos) ((bpos) + PADSZ((bpos),16))
#define PAD_64(bpos) ((bpos) + PADSZ((bpos),64))
#define PAD_4096(bpos) ((bpos) + PADSZ((bpos),4096))

/* Diagnostic functions */
#ifdef ZLIB_DEBUG
   extern int Z_INTERNAL z_verbose;
   extern void Z_INTERNAL z_error(const char *m);
#  define Assert(cond, msg) {int _cond = (cond); if (!_cond) z_error(msg);}
#  define Trace(x) {if (z_verbose >= 0) fprintf x;}
#  define Tracev(x) {if (z_verbose > 0) fprintf x;}
#  define Tracevv(x) {if (z_verbose > 1) fprintf x;}
#  define Tracec(c, x) {if (z_verbose > 0 && (c)) fprintf x;}
#  define Tracecv(c, x) {if (z_verbose > 1 && (c)) fprintf x;}
#else
#  define Assert(cond, msg)
#  define Trace(x)
#  define Tracev(x)
#  define Tracevv(x)
#  define Tracec(c, x)
#  define Tracecv(c, x)
#endif

/* OPTIMAL_CMP values determine the comparison width:
 * 64: Best for 64-bit architectures with unaligned access
 * 32: Best for 32-bit architectures with unaligned access
 * 16: Safe default for unknown architectures
 * 8:  Safe fallback for architectures without unaligned access
 * Note: The unaligned access mentioned is cpu-support, this allows compiler or
 *       separate unaligned intrinsics to utilize safe unaligned access, without
 *       utilizing unaligned C pointers that are known to have undefined behavior.
 */
#if !defined(OPTIMAL_CMP)
#  if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__) || defined(_M_AMD64)
#    define OPTIMAL_CMP 64
#  elif defined(__i386__) || defined(__i486__) || defined(__i586__) || \
        defined(__i686__) || defined(_X86_) || defined(_M_IX86)
#    define OPTIMAL_CMP 32
#  elif defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
#    if defined(__ARM_FEATURE_UNALIGNED) || defined(_WIN32)
#      define OPTIMAL_CMP 64
#    else
#      define OPTIMAL_CMP 8
#    endif
#  elif defined(__arm__) || defined(_M_ARM)
#    if defined(__ARM_FEATURE_UNALIGNED) || defined(_WIN32)
#      define OPTIMAL_CMP 32
#    else
#      define OPTIMAL_CMP 8
#    endif
#  elif defined(__powerpc64__) || defined(__ppc64__)
#    define OPTIMAL_CMP 64
#  elif defined(__powerpc__) || defined(__ppc__) || defined(__PPC__)
#    define OPTIMAL_CMP 32
#  endif
#endif
#if !defined(OPTIMAL_CMP)
#  define OPTIMAL_CMP 16
#endif

#if defined(__has_feature)
#  if __has_feature(address_sanitizer)
#    define Z_ADDRESS_SANITIZER 1
#  endif
#elif defined(__SANITIZE_ADDRESS__)
#  define Z_ADDRESS_SANITIZER 1
#endif

/*
 * __asan_loadN() and __asan_storeN() calls are inserted by compilers in order to check memory accesses.
 * They can be called manually too, with the following caveats:
 * gcc says: "warning: implicit declaration of function '...'"
 * g++ says: "error: new declaration '...' ambiguates built-in declaration '...'"
 * Accommodate both.
 */
#ifdef Z_ADDRESS_SANITIZER
#ifndef __cplusplus
void __asan_loadN(void *, long);
void __asan_storeN(void *, long);
#endif
#else
#  define __asan_loadN(a, size) do { Z_UNUSED(a); Z_UNUSED(size); } while (0)
#  define __asan_storeN(a, size) do { Z_UNUSED(a); Z_UNUSED(size); } while (0)
#endif

#if defined(__has_feature)
#  if __has_feature(memory_sanitizer)
#    define Z_MEMORY_SANITIZER 1
#    include <sanitizer/msan_interface.h>
#  endif
#endif

#ifndef Z_MEMORY_SANITIZER
#  define __msan_check_mem_is_initialized(a, size) do { Z_UNUSED(a); Z_UNUSED(size); } while (0)
#  define __msan_unpoison(a, size) do { Z_UNUSED(a); Z_UNUSED(size); } while (0)
#endif

/* Notify sanitizer runtime about an upcoming read access. */
#define instrument_read(a, size) do {             \
    void *__a = (void *)(a);                      \
    long __size = size;                           \
    __asan_loadN(__a, __size);                    \
    __msan_check_mem_is_initialized(__a, __size); \
} while (0)

/* Notify sanitizer runtime about an upcoming write access. */
#define instrument_write(a, size) do { \
   void *__a = (void *)(a);            \
   long __size = size;                 \
   __asan_storeN(__a, __size);         \
} while (0)

/* Notify sanitizer runtime about an upcoming read/write access. */
#define instrument_read_write(a, size) do {       \
    void *__a = (void *)(a);                      \
    long __size = size;                           \
    __asan_storeN(__a, __size);                   \
    __msan_check_mem_is_initialized(__a, __size); \
} while (0)

#endif
