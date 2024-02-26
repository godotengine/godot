/*
 * Copyright 2006 The Android Open Source Project
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkTypes_DEFINED
#define SkTypes_DEFINED

/** \file SkTypes.h
*/

// Pre-SkUserConfig.h setup.

// Allows embedders that want to disable macros that take arguments to just
// define that symbol to be one of these
#define SK_NOTHING_ARG1(arg1)
#define SK_NOTHING_ARG2(arg1, arg2)
#define SK_NOTHING_ARG3(arg1, arg2, arg3)

#if !defined(SK_BUILD_FOR_ANDROID) && !defined(SK_BUILD_FOR_IOS) && !defined(SK_BUILD_FOR_WIN) && \
    !defined(SK_BUILD_FOR_UNIX) && !defined(SK_BUILD_FOR_MAC)

    #ifdef __APPLE__
        #include <TargetConditionals.h>
    #endif

    #if defined(_WIN32) || defined(__SYMBIAN32__)
        #define SK_BUILD_FOR_WIN
    #elif defined(ANDROID) || defined(__ANDROID__)
        #define SK_BUILD_FOR_ANDROID
    #elif defined(linux) || defined(__linux) || defined(__FreeBSD__) || \
          defined(__OpenBSD__) || defined(__sun) || defined(__NetBSD__) || \
          defined(__DragonFly__) || defined(__Fuchsia__) || \
          defined(__GLIBC__) || defined(__GNU__) || defined(__unix__)
        #define SK_BUILD_FOR_UNIX
    #elif TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR
        #define SK_BUILD_FOR_IOS
    #else
        #define SK_BUILD_FOR_MAC
    #endif

#endif

#if defined(SK_BUILD_FOR_WIN) && !defined(__clang__)
    #if !defined(SK_RESTRICT)
        #define SK_RESTRICT __restrict
    #endif
    #if !defined(SK_WARN_UNUSED_RESULT)
        #define SK_WARN_UNUSED_RESULT
    #endif
#endif

#if !defined(SK_RESTRICT)
    #define SK_RESTRICT __restrict__
#endif

#if !defined(SK_WARN_UNUSED_RESULT)
    #define SK_WARN_UNUSED_RESULT __attribute__((warn_unused_result))
#endif

#if !defined(SK_CPU_BENDIAN) && !defined(SK_CPU_LENDIAN)
    #if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
        #define SK_CPU_BENDIAN
    #elif defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
        #define SK_CPU_LENDIAN
    #elif defined(__sparc) || defined(__sparc__) || \
      defined(_POWER) || defined(__powerpc__) || \
      defined(__ppc__) || defined(__hppa) || \
      defined(__PPC__) || defined(__PPC64__) || \
      defined(_MIPSEB) || defined(__ARMEB__) || \
      defined(__s390__) || \
      (defined(__sh__) && defined(__BIG_ENDIAN__)) || \
      (defined(__ia64) && defined(__BIG_ENDIAN__))
         #define SK_CPU_BENDIAN
    #else
        #define SK_CPU_LENDIAN
    #endif
#endif

#if defined(__i386) || defined(_M_IX86) ||  defined(__x86_64__) || defined(_M_X64)
  #define SK_CPU_X86 1
#endif

/**
 *  SK_CPU_SSE_LEVEL
 *
 *  If defined, SK_CPU_SSE_LEVEL should be set to the highest supported level.
 *  On non-intel CPU this should be undefined.
 */
#define SK_CPU_SSE_LEVEL_SSE1     10
#define SK_CPU_SSE_LEVEL_SSE2     20
#define SK_CPU_SSE_LEVEL_SSE3     30
#define SK_CPU_SSE_LEVEL_SSSE3    31
#define SK_CPU_SSE_LEVEL_SSE41    41
#define SK_CPU_SSE_LEVEL_SSE42    42
#define SK_CPU_SSE_LEVEL_AVX      51
#define SK_CPU_SSE_LEVEL_AVX2     52
#define SK_CPU_SSE_LEVEL_SKX      60

// Are we in GCC/Clang?
#ifndef SK_CPU_SSE_LEVEL
    // These checks must be done in descending order to ensure we set the highest
    // available SSE level.
    #if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512CD__) && \
        defined(__AVX512BW__) && defined(__AVX512VL__)
        #define SK_CPU_SSE_LEVEL    SK_CPU_SSE_LEVEL_SKX
    #elif defined(__AVX2__)
        #define SK_CPU_SSE_LEVEL    SK_CPU_SSE_LEVEL_AVX2
    #elif defined(__AVX__)
        #define SK_CPU_SSE_LEVEL    SK_CPU_SSE_LEVEL_AVX
    #elif defined(__SSE4_2__)
        #define SK_CPU_SSE_LEVEL    SK_CPU_SSE_LEVEL_SSE42
    #elif defined(__SSE4_1__)
        #define SK_CPU_SSE_LEVEL    SK_CPU_SSE_LEVEL_SSE41
    #elif defined(__SSSE3__)
        #define SK_CPU_SSE_LEVEL    SK_CPU_SSE_LEVEL_SSSE3
    #elif defined(__SSE3__)
        #define SK_CPU_SSE_LEVEL    SK_CPU_SSE_LEVEL_SSE3
    #elif defined(__SSE2__)
        #define SK_CPU_SSE_LEVEL    SK_CPU_SSE_LEVEL_SSE2
    #endif
#endif

// Are we in VisualStudio?
#ifndef SK_CPU_SSE_LEVEL
    // These checks must be done in descending order to ensure we set the highest
    // available SSE level. 64-bit intel guarantees at least SSE2 support.
    #if defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512CD__) && \
        defined(__AVX512BW__) && defined(__AVX512VL__)
        #define SK_CPU_SSE_LEVEL        SK_CPU_SSE_LEVEL_SKX
    #elif defined(__AVX2__)
        #define SK_CPU_SSE_LEVEL        SK_CPU_SSE_LEVEL_AVX2
    #elif defined(__AVX__)
        #define SK_CPU_SSE_LEVEL        SK_CPU_SSE_LEVEL_AVX
    #elif defined(_M_X64) || defined(_M_AMD64)
        #define SK_CPU_SSE_LEVEL        SK_CPU_SSE_LEVEL_SSE2
    #elif defined(_M_IX86_FP)
        #if _M_IX86_FP >= 2
            #define SK_CPU_SSE_LEVEL    SK_CPU_SSE_LEVEL_SSE2
        #elif _M_IX86_FP == 1
            #define SK_CPU_SSE_LEVEL    SK_CPU_SSE_LEVEL_SSE1
        #endif
    #endif
#endif

// ARM defines
#if defined(__arm__) && (!defined(__APPLE__) || !TARGET_IPHONE_SIMULATOR)
    #define SK_CPU_ARM32
#elif defined(__aarch64__)
    #define SK_CPU_ARM64
#endif

// All 64-bit ARM chips have NEON.  Many 32-bit ARM chips do too.
#if !defined(SK_ARM_HAS_NEON) && defined(__ARM_NEON)
    #define SK_ARM_HAS_NEON
#endif

#if defined(__ARM_FEATURE_CRC32)
    #define SK_ARM_HAS_CRC32
#endif


// DLL/.so exports.
#if !defined(SKIA_IMPLEMENTATION)
    #define SKIA_IMPLEMENTATION 0
#endif
#if !defined(SK_API)
    #if defined(SKIA_DLL)
        #if defined(_MSC_VER)
            #if SKIA_IMPLEMENTATION
                #define SK_API __declspec(dllexport)
            #else
                #define SK_API __declspec(dllimport)
            #endif
        #else
            #define SK_API __attribute__((visibility("default")))
        #endif
    #else
        #define SK_API
    #endif
#endif

// SK_SPI is functionally identical to SK_API, but used within src to clarify that it's less stable
#if !defined(SK_SPI)
    #define SK_SPI SK_API
#endif

// IWYU pragma: begin_exports
#if defined (SK_USER_CONFIG_HEADER)
    #include SK_USER_CONFIG_HEADER
#else
    #include "include/config/SkUserConfig.h"
#endif
#include <stddef.h>
#include <stdint.h>
// IWYU pragma: end_exports

// Post SkUserConfig.h checks and such.
#if !defined(SK_DEBUG) && !defined(SK_RELEASE)
    #ifdef NDEBUG
        #define SK_RELEASE
    #else
        #define SK_DEBUG
    #endif
#endif

#if defined(SK_DEBUG) && defined(SK_RELEASE)
#  error "cannot define both SK_DEBUG and SK_RELEASE"
#elif !defined(SK_DEBUG) && !defined(SK_RELEASE)
#  error "must define either SK_DEBUG or SK_RELEASE"
#endif

#if defined(SK_CPU_LENDIAN) && defined(SK_CPU_BENDIAN)
#  error "cannot define both SK_CPU_LENDIAN and SK_CPU_BENDIAN"
#elif !defined(SK_CPU_LENDIAN) && !defined(SK_CPU_BENDIAN)
#  error "must define either SK_CPU_LENDIAN or SK_CPU_BENDIAN"
#endif

#if defined(SK_CPU_BENDIAN) && !defined(I_ACKNOWLEDGE_SKIA_DOES_NOT_SUPPORT_BIG_ENDIAN)
    #error "The Skia team is not endian-savvy enough to support big-endian CPUs."
    #error "If you still want to use Skia,"
    #error "please define I_ACKNOWLEDGE_SKIA_DOES_NOT_SUPPORT_BIG_ENDIAN."
#endif

#if !defined(SK_ATTRIBUTE)
#  if defined(__clang__) || defined(__GNUC__)
#    define SK_ATTRIBUTE(attr) __attribute__((attr))
#  else
#    define SK_ATTRIBUTE(attr)
#  endif
#endif

#if !defined(SK_SUPPORT_GPU)
#  define SK_SUPPORT_GPU 1
#endif

#if SK_SUPPORT_GPU || defined(SK_GRAPHITE_ENABLED)
#  if !defined(SK_ENABLE_SKSL)
#    define SK_ENABLE_SKSL
#  endif
#else
#  undef SK_GL
#  undef SK_VULKAN
#  undef SK_METAL
#  undef SK_DAWN
#  undef SK_DIRECT3D
#endif

#if !defined(SkUNREACHABLE)
#  if defined(_MSC_VER) && !defined(__clang__)
#    include <intrin.h>
#    define FAST_FAIL_INVALID_ARG                 5
// See https://developercommunity.visualstudio.com/content/problem/1128631/code-flow-doesnt-see-noreturn-with-extern-c.html
// for why this is wrapped. Hopefully removable after msvc++ 19.27 is no longer supported.
[[noreturn]] static inline void sk_fast_fail() { __fastfail(FAST_FAIL_INVALID_ARG); }
#    define SkUNREACHABLE sk_fast_fail()
#  else
#    define SkUNREACHABLE __builtin_trap()
#  endif
#endif

#if defined(SK_BUILD_FOR_GOOGLE3)
    void SkDebugfForDumpStackTrace(const char* data, void* unused);
    namespace base {
        void DumpStackTrace(int skip_count, void w(const char*, void*), void* arg);
    }
#  define SK_DUMP_GOOGLE3_STACK() ::base::DumpStackTrace(0, SkDebugfForDumpStackTrace, nullptr)
#else
#  define SK_DUMP_GOOGLE3_STACK()
#endif

#ifndef SK_ABORT
#  ifdef SK_BUILD_FOR_WIN
     // This style lets Visual Studio follow errors back to the source file.
#    define SK_DUMP_LINE_FORMAT "%s(%d)"
#  else
#    define SK_DUMP_LINE_FORMAT "%s:%d"
#  endif
#  define SK_ABORT(message, ...) \
    do { \
        SkDebugf(SK_DUMP_LINE_FORMAT ": fatal error: \"" message "\"\n", \
                 __FILE__, __LINE__, ##__VA_ARGS__); \
        SK_DUMP_GOOGLE3_STACK(); \
        sk_abort_no_print(); \
    } while (false)
#endif

// If SK_R32_SHIFT is set, we'll use that to choose RGBA or BGRA.
// If not, we'll default to RGBA everywhere except BGRA on Windows.
#if defined(SK_R32_SHIFT)
    static_assert(SK_R32_SHIFT == 0 || SK_R32_SHIFT == 16, "");
#elif defined(SK_BUILD_FOR_WIN)
    #define SK_R32_SHIFT 16
#else
    #define SK_R32_SHIFT 0
#endif

#if defined(SK_B32_SHIFT)
    static_assert(SK_B32_SHIFT == (16-SK_R32_SHIFT), "");
#else
    #define SK_B32_SHIFT (16-SK_R32_SHIFT)
#endif

#define SK_G32_SHIFT 8
#define SK_A32_SHIFT 24


/**
 * SK_PMCOLOR_BYTE_ORDER can be used to query the byte order of SkPMColor at compile time.
 */
#ifdef SK_CPU_BENDIAN
#  define SK_PMCOLOR_BYTE_ORDER(C0, C1, C2, C3)     \
        (SK_ ## C3 ## 32_SHIFT == 0  &&             \
         SK_ ## C2 ## 32_SHIFT == 8  &&             \
         SK_ ## C1 ## 32_SHIFT == 16 &&             \
         SK_ ## C0 ## 32_SHIFT == 24)
#else
#  define SK_PMCOLOR_BYTE_ORDER(C0, C1, C2, C3)     \
        (SK_ ## C0 ## 32_SHIFT == 0  &&             \
         SK_ ## C1 ## 32_SHIFT == 8  &&             \
         SK_ ## C2 ## 32_SHIFT == 16 &&             \
         SK_ ## C3 ## 32_SHIFT == 24)
#endif

#if defined SK_DEBUG && defined SK_BUILD_FOR_WIN
    #ifdef free
        #undef free
    #endif
    #include <crtdbg.h>
    #undef free
#endif

#if !defined(SK_UNUSED)
#  if !defined(__clang__) && defined(_MSC_VER)
#    define SK_UNUSED __pragma(warning(suppress:4189))
#  else
#    define SK_UNUSED SK_ATTRIBUTE(unused)
#  endif
#endif

#if !defined(SK_MAYBE_UNUSED)
#  if defined(__clang__) || defined(__GNUC__)
#    define SK_MAYBE_UNUSED [[maybe_unused]]
#  else
#    define SK_MAYBE_UNUSED
#  endif
#endif

/**
 * If your judgment is better than the compiler's (i.e. you've profiled it),
 * you can use SK_ALWAYS_INLINE to force inlining. E.g.
 *     inline void someMethod() { ... }             // may not be inlined
 *     SK_ALWAYS_INLINE void someMethod() { ... }   // should always be inlined
 */
#if !defined(SK_ALWAYS_INLINE)
#  if defined(SK_BUILD_FOR_WIN)
#    define SK_ALWAYS_INLINE __forceinline
#  else
#    define SK_ALWAYS_INLINE SK_ATTRIBUTE(always_inline) inline
#  endif
#endif

/**
 * If your judgment is better than the compiler's (i.e. you've profiled it),
 * you can use SK_NEVER_INLINE to prevent inlining.
 */
#if !defined(SK_NEVER_INLINE)
#  if defined(SK_BUILD_FOR_WIN)
#    define SK_NEVER_INLINE __declspec(noinline)
#  else
#    define SK_NEVER_INLINE SK_ATTRIBUTE(noinline)
#  endif
#endif

#ifndef SK_PRINTF_LIKE
#  if defined(__clang__) || defined(__GNUC__)
#    define SK_PRINTF_LIKE(A, B) __attribute__((format(printf, (A), (B))))
#  else
#    define SK_PRINTF_LIKE(A, B)
#  endif
#endif

#ifndef SK_ALLOW_STATIC_GLOBAL_INITIALIZERS
    #define SK_ALLOW_STATIC_GLOBAL_INITIALIZERS 0
#endif

#if !defined(SK_GAMMA_EXPONENT)
    #define SK_GAMMA_EXPONENT (0.0f)  // SRGB
#endif

#ifndef GR_TEST_UTILS
#  define GR_TEST_UTILS 0
#endif

#if !SK_SUPPORT_GPU
   #define SK_GPU_V1 0 // always false if Ganesh is disabled
#elif !defined(SK_GPU_V1)
#  define SK_GPU_V1 1   // otherwise default to v1 enabled
#endif

#if defined(SK_HISTOGRAM_ENUMERATION)  || \
    defined(SK_HISTOGRAM_BOOLEAN)      || \
    defined(SK_HISTOGRAM_EXACT_LINEAR) || \
    defined(SK_HISTOGRAM_MEMORY_KB)
#  define SK_HISTOGRAMS_ENABLED 1
#else
#  define SK_HISTOGRAMS_ENABLED 0
#endif

#ifndef SK_HISTOGRAM_BOOLEAN
#  define SK_HISTOGRAM_BOOLEAN(name, sample)
#endif

#ifndef SK_HISTOGRAM_ENUMERATION
#  define SK_HISTOGRAM_ENUMERATION(name, sample, enum_size)
#endif

#ifndef SK_HISTOGRAM_EXACT_LINEAR
#  define SK_HISTOGRAM_EXACT_LINEAR(name, sample, value_max)
#endif

#ifndef SK_HISTOGRAM_MEMORY_KB
#  define SK_HISTOGRAM_MEMORY_KB(name, sample)
#endif

#define SK_HISTOGRAM_PERCENTAGE(name, percent_as_int) \
    SK_HISTOGRAM_EXACT_LINEAR(name, percent_as_int, 101)

#ifndef SK_DISABLE_LEGACY_SHADERCONTEXT
#define SK_ENABLE_LEGACY_SHADERCONTEXT
#endif

#ifdef SK_ENABLE_API_AVAILABLE
#define SK_API_AVAILABLE API_AVAILABLE
#else
#define SK_API_AVAILABLE(...)
#endif

#if defined(SK_BUILD_FOR_LIBFUZZER) || defined(SK_BUILD_FOR_AFL_FUZZ)
    #define SK_BUILD_FOR_FUZZER
#endif

/** Called internally if we hit an unrecoverable error.
    The platform implementation must not return, but should either throw
    an exception or otherwise exit.
*/
[[noreturn]] SK_API extern void sk_abort_no_print(void);

#ifndef SkDebugf
    SK_API void SkDebugf(const char format[], ...) SK_PRINTF_LIKE(1, 2);
#endif

// SkASSERT, SkASSERTF and SkASSERT_RELEASE can be used as stand alone assertion expressions, e.g.
//    uint32_t foo(int x) {
//        SkASSERT(x > 4);
//        return x - 4;
//    }
// and are also written to be compatible with constexpr functions:
//    constexpr uint32_t foo(int x) {
//        return SkASSERT(x > 4),
//               x - 4;
//    }
#define SkASSERT_RELEASE(cond) \
        static_cast<void>( (cond) ? (void)0 : []{ SK_ABORT("assert(%s)", #cond); }() )

#ifdef SK_DEBUG
    #define SkASSERT(cond) SkASSERT_RELEASE(cond)
    #define SkASSERTF(cond, fmt, ...) static_cast<void>( (cond) ? (void)0 : [&]{ \
                                          SkDebugf(fmt"\n", ##__VA_ARGS__);      \
                                          SK_ABORT("assert(%s)", #cond);         \
                                      }() )
    #define SkDEBUGFAIL(message)        SK_ABORT("%s", message)
    #define SkDEBUGFAILF(fmt, ...)      SK_ABORT(fmt, ##__VA_ARGS__)
    #define SkDEBUGCODE(...)            __VA_ARGS__
    #define SkDEBUGF(...)               SkDebugf(__VA_ARGS__)
    #define SkAssertResult(cond)        SkASSERT(cond)
#else
    #define SkASSERT(cond)            static_cast<void>(0)
    #define SkASSERTF(cond, fmt, ...) static_cast<void>(0)
    #define SkDEBUGFAIL(message)
    #define SkDEBUGFAILF(fmt, ...)
    #define SkDEBUGCODE(...)
    #define SkDEBUGF(...)

    // unlike SkASSERT, this macro executes its condition in the non-debug build.
    // The if is present so that this can be used with functions marked SK_WARN_UNUSED_RESULT.
    #define SkAssertResult(cond)         if (cond) {} do {} while(false)
#endif

////////////////////////////////////////////////////////////////////////////////

/** Fast type for unsigned 8 bits. Use for parameter passing and local
    variables, not for storage
*/
typedef unsigned U8CPU;

/** Fast type for unsigned 16 bits. Use for parameter passing and local
    variables, not for storage
*/
typedef unsigned U16CPU;

/** @return false or true based on the condition
*/
template <typename T> static constexpr bool SkToBool(const T& x) {
    return (bool)x;
}

static constexpr int16_t SK_MaxS16 = INT16_MAX;
static constexpr int16_t SK_MinS16 = -SK_MaxS16;

static constexpr int32_t SK_MaxS32 = INT32_MAX;
static constexpr int32_t SK_MinS32 = -SK_MaxS32;
static constexpr int32_t SK_NaN32  = INT32_MIN;

static constexpr int64_t SK_MaxS64 = INT64_MAX;
static constexpr int64_t SK_MinS64 = -SK_MaxS64;

static inline constexpr int32_t SkLeftShift(int32_t value, int32_t shift) {
    return (int32_t) ((uint32_t) value << shift);
}

static inline constexpr int64_t SkLeftShift(int64_t value, int32_t shift) {
    return (int64_t) ((uint64_t) value << shift);
}

////////////////////////////////////////////////////////////////////////////////

/** @return the number of entries in an array (not a pointer)
*/
template <typename T, size_t N> char (&SkArrayCountHelper(T (&array)[N]))[N];
#define SK_ARRAY_COUNT(array) (sizeof(SkArrayCountHelper(array)))

////////////////////////////////////////////////////////////////////////////////

template <typename T> static constexpr T SkAlign2(T x) { return (x + 1) >> 1 << 1; }
template <typename T> static constexpr T SkAlign4(T x) { return (x + 3) >> 2 << 2; }
template <typename T> static constexpr T SkAlign8(T x) { return (x + 7) >> 3 << 3; }

template <typename T> static constexpr bool SkIsAlign2(T x) { return 0 == (x & 1); }
template <typename T> static constexpr bool SkIsAlign4(T x) { return 0 == (x & 3); }
template <typename T> static constexpr bool SkIsAlign8(T x) { return 0 == (x & 7); }

template <typename T> static constexpr T SkAlignPtr(T x) {
    return sizeof(void*) == 8 ? SkAlign8(x) : SkAlign4(x);
}
template <typename T> static constexpr bool SkIsAlignPtr(T x) {
    return sizeof(void*) == 8 ? SkIsAlign8(x) : SkIsAlign4(x);
}

/**
 *  align up to a power of 2
 */
static inline constexpr size_t SkAlignTo(size_t x, size_t alignment) {
    // The same as alignment && SkIsPow2(value), w/o a dependency cycle.
    SkASSERT(alignment && (alignment & (alignment - 1)) == 0);
    return (x + alignment - 1) & ~(alignment - 1);
}

typedef uint32_t SkFourByteTag;
static inline constexpr SkFourByteTag SkSetFourByteTag(char a, char b, char c, char d) {
    return (((uint32_t)a << 24) | ((uint32_t)b << 16) | ((uint32_t)c << 8) | (uint32_t)d);
}

////////////////////////////////////////////////////////////////////////////////

/** 32 bit integer to hold a unicode value
*/
typedef int32_t SkUnichar;

/** 16 bit unsigned integer to hold a glyph index
*/
typedef uint16_t SkGlyphID;

/** 32 bit value to hold a millisecond duration
    Note that SK_MSecMax is about 25 days.
*/
typedef uint32_t SkMSec;

/** Maximum representable milliseconds; 24d 20h 31m 23.647s.
*/
static constexpr SkMSec SK_MSecMax = INT32_MAX;

/** The generation IDs in Skia reserve 0 has an invalid marker.
*/
static constexpr uint32_t SK_InvalidGenID = 0;

/** The unique IDs in Skia reserve 0 has an invalid marker.
*/
static constexpr uint32_t SK_InvalidUniqueID = 0;

static inline int32_t SkAbs32(int32_t value) {
    SkASSERT(value != SK_NaN32);  // The most negative int32_t can't be negated.
    if (value < 0) {
        value = -value;
    }
    return value;
}

template <typename T> static inline T SkTAbs(T value) {
    if (value < 0) {
        value = -value;
    }
    return value;
}

////////////////////////////////////////////////////////////////////////////////

/** Indicates whether an allocation should count against a cache budget.
*/
enum class SkBudgeted : bool {
    kNo  = false,
    kYes = true
};

/** Indicates whether a backing store needs to be an exact match or can be
    larger than is strictly necessary
*/
enum class SkBackingFit {
    kApprox,
    kExact
};

#endif
