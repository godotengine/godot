/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

/**
 * # CategoryEndian
 *
 * Functions converting endian-specific values to different byte orders.
 *
 * These functions either unconditionally swap byte order (SDL_Swap16,
 * SDL_Swap32, SDL_Swap64, SDL_SwapFloat), or they swap to/from the system's
 * native byte order (SDL_Swap16LE, SDL_Swap16BE, SDL_Swap32LE, SDL_Swap32BE,
 * SDL_Swap32LE, SDL_Swap32BE, SDL_SwapFloatLE, SDL_SwapFloatBE). In the
 * latter case, the functionality is provided by macros that become no-ops if
 * a swap isn't necessary: on an x86 (littleendian) processor, SDL_Swap32LE
 * does nothing, but SDL_Swap32BE reverses the bytes of the data. On a PowerPC
 * processor (bigendian), the macros behavior is reversed.
 *
 * The swap routines are inline functions, and attempt to use compiler
 * intrinsics, inline assembly, and other magic to make byteswapping
 * efficient.
 */

#ifndef SDL_endian_h_
#define SDL_endian_h_

#include <SDL3/SDL_stdinc.h>

#if defined(_MSC_VER) && (_MSC_VER >= 1400)
/* As of Clang 11, '_m_prefetchw' is conflicting with the winnt.h's version,
   so we define the needed '_m_prefetch' here as a pseudo-header, until the issue is fixed. */
#ifdef __clang__
#ifndef __PRFCHWINTRIN_H
#define __PRFCHWINTRIN_H
static __inline__ void __attribute__((__always_inline__, __nodebug__))
_m_prefetch(void *__P)
{
  __builtin_prefetch(__P, 0, 3 /* _MM_HINT_T0 */);
}
#endif /* __PRFCHWINTRIN_H */
#endif /* __clang__ */

#include <intrin.h>
#endif

/**
 *  \name The two types of endianness
 */
/* @{ */


/**
 * A value to represent littleendian byteorder.
 *
 * This is used with the preprocessor macro SDL_BYTEORDER, to determine a
 * platform's byte ordering:
 *
 * ```c
 * #if SDL_BYTEORDER == SDL_LIL_ENDIAN
 * SDL_Log("This system is littleendian.");
 * #endif
 * ```
 *
 * \since This macro is available since SDL 3.2.0.
 *
 * \sa SDL_BYTEORDER
 * \sa SDL_BIG_ENDIAN
 */
#define SDL_LIL_ENDIAN  1234

/**
 * A value to represent bigendian byteorder.
 *
 * This is used with the preprocessor macro SDL_BYTEORDER, to determine a
 * platform's byte ordering:
 *
 * ```c
 * #if SDL_BYTEORDER == SDL_BIG_ENDIAN
 * SDL_Log("This system is bigendian.");
 * #endif
 * ```
 *
 * \since This macro is available since SDL 3.2.0.
 *
 * \sa SDL_BYTEORDER
 * \sa SDL_LIL_ENDIAN
 */
#define SDL_BIG_ENDIAN  4321

/* @} */

#ifndef SDL_BYTEORDER
#ifdef SDL_WIKI_DOCUMENTATION_SECTION

/**
 * A macro that reports the target system's byte order.
 *
 * This is set to either SDL_LIL_ENDIAN or SDL_BIG_ENDIAN (and maybe other
 * values in the future, if something else becomes popular). This can be
 * tested with the preprocessor, so decisions can be made at compile time.
 *
 * ```c
 * #if SDL_BYTEORDER == SDL_BIG_ENDIAN
 * SDL_Log("This system is bigendian.");
 * #endif
 * ```
 *
 * \since This macro is available since SDL 3.2.0.
 *
 * \sa SDL_LIL_ENDIAN
 * \sa SDL_BIG_ENDIAN
 */
#define SDL_BYTEORDER   SDL_LIL_ENDIAN___or_maybe___SDL_BIG_ENDIAN
#elif defined(SDL_PLATFORM_LINUX)
#include <endian.h>
#define SDL_BYTEORDER  __BYTE_ORDER
#elif defined(SDL_PLATFORM_SOLARIS)
#include <sys/byteorder.h>
#if defined(_LITTLE_ENDIAN)
#define SDL_BYTEORDER   SDL_LIL_ENDIAN
#elif defined(_BIG_ENDIAN)
#define SDL_BYTEORDER   SDL_BIG_ENDIAN
#else
#error Unsupported endianness
#endif
#elif defined(SDL_PLATFORM_OPENBSD) || defined(__DragonFly__)
#include <endian.h>
#define SDL_BYTEORDER  BYTE_ORDER
#elif defined(SDL_PLATFORM_FREEBSD) || defined(SDL_PLATFORM_NETBSD)
#include <sys/endian.h>
#define SDL_BYTEORDER  BYTE_ORDER
/* predefs from newer gcc and clang versions: */
#elif defined(__ORDER_LITTLE_ENDIAN__) && defined(__ORDER_BIG_ENDIAN__) && defined(__BYTE_ORDER__)
#if (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define SDL_BYTEORDER   SDL_LIL_ENDIAN
#elif (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#define SDL_BYTEORDER   SDL_BIG_ENDIAN
#else
#error Unsupported endianness
#endif /**/
#else
#if defined(__hppa__) || \
    defined(__m68k__) || defined(mc68000) || defined(_M_M68K) || \
    (defined(__MIPS__) && defined(__MIPSEB__)) || \
    defined(__ppc__) || defined(__POWERPC__) || defined(__powerpc__) || defined(__PPC__) || \
    defined(__sparc__) || defined(__sparc)
#define SDL_BYTEORDER   SDL_BIG_ENDIAN
#else
#define SDL_BYTEORDER   SDL_LIL_ENDIAN
#endif
#endif /* SDL_PLATFORM_LINUX */
#endif /* !SDL_BYTEORDER */

#ifndef SDL_FLOATWORDORDER
#ifdef SDL_WIKI_DOCUMENTATION_SECTION

/**
 * A macro that reports the target system's floating point word order.
 *
 * This is set to either SDL_LIL_ENDIAN or SDL_BIG_ENDIAN (and maybe other
 * values in the future, if something else becomes popular). This can be
 * tested with the preprocessor, so decisions can be made at compile time.
 *
 * ```c
 * #if SDL_FLOATWORDORDER == SDL_BIG_ENDIAN
 * SDL_Log("This system's floats are bigendian.");
 * #endif
 * ```
 *
 * \since This macro is available since SDL 3.2.0.
 *
 * \sa SDL_LIL_ENDIAN
 * \sa SDL_BIG_ENDIAN
 */
#define SDL_FLOATWORDORDER   SDL_LIL_ENDIAN___or_maybe___SDL_BIG_ENDIAN
/* predefs from newer gcc versions: */
#elif defined(__ORDER_LITTLE_ENDIAN__) && defined(__ORDER_BIG_ENDIAN__) && defined(__FLOAT_WORD_ORDER__)
#if (__FLOAT_WORD_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define SDL_FLOATWORDORDER   SDL_LIL_ENDIAN
#elif (__FLOAT_WORD_ORDER__ == __ORDER_BIG_ENDIAN__)
#define SDL_FLOATWORDORDER   SDL_BIG_ENDIAN
#else
#error Unsupported endianness
#endif /**/
#elif defined(__MAVERICK__)
/* For Maverick, float words are always little-endian. */
#define SDL_FLOATWORDORDER   SDL_LIL_ENDIAN
#elif (defined(__arm__) || defined(__thumb__)) && !defined(__VFP_FP__) && !defined(__ARM_EABI__)
/* For FPA, float words are always big-endian. */
#define SDL_FLOATWORDORDER   SDL_BIG_ENDIAN
#else
/* By default, assume that floats words follow the memory system mode. */
#define SDL_FLOATWORDORDER   SDL_BYTEORDER
#endif /* __FLOAT_WORD_ORDER__ */
#endif /* !SDL_FLOATWORDORDER */


#include <SDL3/SDL_begin_code.h>
/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif

/* various modern compilers may have builtin swap */
#if defined(__GNUC__) || defined(__clang__)
#   define HAS_BUILTIN_BSWAP16 (SDL_HAS_BUILTIN(__builtin_bswap16)) || \
        (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8))
#   define HAS_BUILTIN_BSWAP32 (SDL_HAS_BUILTIN(__builtin_bswap32)) || \
        (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))
#   define HAS_BUILTIN_BSWAP64 (SDL_HAS_BUILTIN(__builtin_bswap64)) || \
        (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))

    /* this one is broken */
#   define HAS_BROKEN_BSWAP (__GNUC__ == 2 && __GNUC_MINOR__ <= 95)
#else
#   define HAS_BUILTIN_BSWAP16 0
#   define HAS_BUILTIN_BSWAP32 0
#   define HAS_BUILTIN_BSWAP64 0
#   define HAS_BROKEN_BSWAP 0
#endif

/* Byte swap 16-bit integer. */
#ifndef SDL_WIKI_DOCUMENTATION_SECTION
#if HAS_BUILTIN_BSWAP16
#define SDL_Swap16(x) __builtin_bswap16(x)
#elif (defined(_MSC_VER) && (_MSC_VER >= 1400)) && !defined(__ICL)
#pragma intrinsic(_byteswap_ushort)
#define SDL_Swap16(x) _byteswap_ushort(x)
#elif defined(__i386__) && !HAS_BROKEN_BSWAP
SDL_FORCE_INLINE Uint16 SDL_Swap16(Uint16 x)
{
  __asm__("xchgb %b0,%h0": "=q"(x):"0"(x));
    return x;
}
#elif defined(__x86_64__)
SDL_FORCE_INLINE Uint16 SDL_Swap16(Uint16 x)
{
  __asm__("xchgb %b0,%h0": "=Q"(x):"0"(x));
    return x;
}
#elif (defined(__powerpc__) || defined(__ppc__))
SDL_FORCE_INLINE Uint16 SDL_Swap16(Uint16 x)
{
    int result;

  __asm__("rlwimi %0,%2,8,16,23": "=&r"(result):"0"(x >> 8), "r"(x));
    return (Uint16)result;
}
#elif (defined(__m68k__) && !defined(__mcoldfire__))
SDL_FORCE_INLINE Uint16 SDL_Swap16(Uint16 x)
{
  __asm__("rorw #8,%0": "=d"(x): "0"(x):"cc");
    return x;
}
#elif defined(__WATCOMC__) && defined(__386__)
extern __inline Uint16 SDL_Swap16(Uint16);
#pragma aux SDL_Swap16 = \
  "xchg al, ah" \
  parm   [ax]   \
  modify [ax];
#else
SDL_FORCE_INLINE Uint16 SDL_Swap16(Uint16 x)
{
    return SDL_static_cast(Uint16, ((x << 8) | (x >> 8)));
}
#endif
#endif

/* Byte swap 32-bit integer. */
#ifndef SDL_WIKI_DOCUMENTATION_SECTION
#if HAS_BUILTIN_BSWAP32
#define SDL_Swap32(x) __builtin_bswap32(x)
#elif (defined(_MSC_VER) && (_MSC_VER >= 1400)) && !defined(__ICL)
#pragma intrinsic(_byteswap_ulong)
#define SDL_Swap32(x) _byteswap_ulong(x)
#elif defined(__i386__) && !HAS_BROKEN_BSWAP
SDL_FORCE_INLINE Uint32 SDL_Swap32(Uint32 x)
{
  __asm__("bswap %0": "=r"(x):"0"(x));
    return x;
}
#elif defined(__x86_64__)
SDL_FORCE_INLINE Uint32 SDL_Swap32(Uint32 x)
{
  __asm__("bswapl %0": "=r"(x):"0"(x));
    return x;
}
#elif (defined(__powerpc__) || defined(__ppc__))
SDL_FORCE_INLINE Uint32 SDL_Swap32(Uint32 x)
{
    Uint32 result;

  __asm__("rlwimi %0,%2,24,16,23": "=&r"(result): "0" (x>>24),  "r"(x));
  __asm__("rlwimi %0,%2,8,8,15"  : "=&r"(result): "0" (result), "r"(x));
  __asm__("rlwimi %0,%2,24,0,7"  : "=&r"(result): "0" (result), "r"(x));
    return result;
}
#elif (defined(__m68k__) && !defined(__mcoldfire__))
SDL_FORCE_INLINE Uint32 SDL_Swap32(Uint32 x)
{
  __asm__("rorw #8,%0\n\tswap %0\n\trorw #8,%0": "=d"(x): "0"(x):"cc");
    return x;
}
#elif defined(__WATCOMC__) && defined(__386__)
extern __inline Uint32 SDL_Swap32(Uint32);
#pragma aux SDL_Swap32 = \
  "bswap eax"  \
  parm   [eax] \
  modify [eax];
#else
SDL_FORCE_INLINE Uint32 SDL_Swap32(Uint32 x)
{
    return SDL_static_cast(Uint32, ((x << 24) | ((x << 8) & 0x00FF0000) |
                                    ((x >> 8) & 0x0000FF00) | (x >> 24)));
}
#endif
#endif

/* Byte swap 64-bit integer. */
#ifndef SDL_WIKI_DOCUMENTATION_SECTION
#if HAS_BUILTIN_BSWAP64
#define SDL_Swap64(x) __builtin_bswap64(x)
#elif (defined(_MSC_VER) && (_MSC_VER >= 1400)) && !defined(__ICL)
#pragma intrinsic(_byteswap_uint64)
#define SDL_Swap64(x) _byteswap_uint64(x)
#elif defined(__i386__) && !HAS_BROKEN_BSWAP
SDL_FORCE_INLINE Uint64 SDL_Swap64(Uint64 x)
{
    union {
        struct {
            Uint32 a, b;
        } s;
        Uint64 u;
    } v;
    v.u = x;
  __asm__("bswapl %0 ; bswapl %1 ; xchgl %0,%1"
          : "=r"(v.s.a), "=r"(v.s.b)
          : "0" (v.s.a),  "1"(v.s.b));
    return v.u;
}
#elif defined(__x86_64__)
SDL_FORCE_INLINE Uint64 SDL_Swap64(Uint64 x)
{
  __asm__("bswapq %0": "=r"(x):"0"(x));
    return x;
}
#elif defined(__WATCOMC__) && defined(__386__)
extern __inline Uint64 SDL_Swap64(Uint64);
#pragma aux SDL_Swap64 = \
  "bswap eax"     \
  "bswap edx"     \
  "xchg eax,edx"  \
  parm [eax edx]  \
  modify [eax edx];
#else
SDL_FORCE_INLINE Uint64 SDL_Swap64(Uint64 x)
{
    Uint32 hi, lo;

    /* Separate into high and low 32-bit values and swap them */
    lo = SDL_static_cast(Uint32, x & 0xFFFFFFFF);
    x >>= 32;
    hi = SDL_static_cast(Uint32, x & 0xFFFFFFFF);
    x = SDL_Swap32(lo);
    x <<= 32;
    x |= SDL_Swap32(hi);
    return (x);
}
#endif
#endif

/**
 * Byte-swap a floating point number.
 *
 * This will always byte-swap the value, whether it's currently in the native
 * byteorder of the system or not. You should use SDL_SwapFloatLE or
 * SDL_SwapFloatBE instead, in most cases.
 *
 * Note that this is a forced-inline function in a header, and not a public
 * API function available in the SDL library (which is to say, the code is
 * embedded in the calling program and the linker and dynamic loader will not
 * be able to find this function inside SDL itself).
 *
 * \param x the value to byte-swap.
 * \returns x, with its bytes in the opposite endian order.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
SDL_FORCE_INLINE float SDL_SwapFloat(float x)
{
    union {
        float f;
        Uint32 ui32;
    } swapper;
    swapper.f = x;
    swapper.ui32 = SDL_Swap32(swapper.ui32);
    return swapper.f;
}

/* remove extra macros */
#undef HAS_BROKEN_BSWAP
#undef HAS_BUILTIN_BSWAP16
#undef HAS_BUILTIN_BSWAP32
#undef HAS_BUILTIN_BSWAP64


#ifdef SDL_WIKI_DOCUMENTATION_SECTION

/**
 * Byte-swap an unsigned 16-bit number.
 *
 * This will always byte-swap the value, whether it's currently in the native
 * byteorder of the system or not. You should use SDL_Swap16LE or SDL_Swap16BE
 * instead, in most cases.
 *
 * Note that this is a forced-inline function in a header, and not a public
 * API function available in the SDL library (which is to say, the code is
 * embedded in the calling program and the linker and dynamic loader will not
 * be able to find this function inside SDL itself).
 *
 * \param x the value to byte-swap.
 * \returns `x`, with its bytes in the opposite endian order.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
SDL_FORCE_INLINE Uint16 SDL_Swap16(Uint16 x) { return x_but_byteswapped; }

/**
 * Byte-swap an unsigned 32-bit number.
 *
 * This will always byte-swap the value, whether it's currently in the native
 * byteorder of the system or not. You should use SDL_Swap32LE or SDL_Swap32BE
 * instead, in most cases.
 *
 * Note that this is a forced-inline function in a header, and not a public
 * API function available in the SDL library (which is to say, the code is
 * embedded in the calling program and the linker and dynamic loader will not
 * be able to find this function inside SDL itself).
 *
 * \param x the value to byte-swap.
 * \returns `x`, with its bytes in the opposite endian order.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
SDL_FORCE_INLINE Uint32 SDL_Swap32(Uint32 x) { return x_but_byteswapped; }

/**
 * Byte-swap an unsigned 64-bit number.
 *
 * This will always byte-swap the value, whether it's currently in the native
 * byteorder of the system or not. You should use SDL_Swap64LE or SDL_Swap64BE
 * instead, in most cases.
 *
 * Note that this is a forced-inline function in a header, and not a public
 * API function available in the SDL library (which is to say, the code is
 * embedded in the calling program and the linker and dynamic loader will not
 * be able to find this function inside SDL itself).
 *
 * \param x the value to byte-swap.
 * \returns `x`, with its bytes in the opposite endian order.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
SDL_FORCE_INLINE Uint32 SDL_Swap64(Uint64 x) { return x_but_byteswapped; }

/**
 * Swap a 16-bit value from littleendian to native byte order.
 *
 * If this is running on a littleendian system, `x` is returned unchanged.
 *
 * This macro never references `x` more than once, avoiding side effects.
 *
 * \param x the value to swap, in littleendian byte order.
 * \returns `x` in native byte order.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_Swap16LE(x) SwapOnlyIfNecessary(x)

/**
 * Swap a 32-bit value from littleendian to native byte order.
 *
 * If this is running on a littleendian system, `x` is returned unchanged.
 *
 * This macro never references `x` more than once, avoiding side effects.
 *
 * \param x the value to swap, in littleendian byte order.
 * \returns `x` in native byte order.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_Swap32LE(x) SwapOnlyIfNecessary(x)

/**
 * Swap a 64-bit value from littleendian to native byte order.
 *
 * If this is running on a littleendian system, `x` is returned unchanged.
 *
 * This macro never references `x` more than once, avoiding side effects.
 *
 * \param x the value to swap, in littleendian byte order.
 * \returns `x` in native byte order.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_Swap64LE(x) SwapOnlyIfNecessary(x)

/**
 * Swap a floating point value from littleendian to native byte order.
 *
 * If this is running on a littleendian system, `x` is returned unchanged.
 *
 * This macro never references `x` more than once, avoiding side effects.
 *
 * \param x the value to swap, in littleendian byte order.
 * \returns `x` in native byte order.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_SwapFloatLE(x) SwapOnlyIfNecessary(x)

/**
 * Swap a 16-bit value from bigendian to native byte order.
 *
 * If this is running on a bigendian system, `x` is returned unchanged.
 *
 * This macro never references `x` more than once, avoiding side effects.
 *
 * \param x the value to swap, in bigendian byte order.
 * \returns `x` in native byte order.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_Swap16BE(x) SwapOnlyIfNecessary(x)

/**
 * Swap a 32-bit value from bigendian to native byte order.
 *
 * If this is running on a bigendian system, `x` is returned unchanged.
 *
 * This macro never references `x` more than once, avoiding side effects.
 *
 * \param x the value to swap, in bigendian byte order.
 * \returns `x` in native byte order.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_Swap32BE(x) SwapOnlyIfNecessary(x)

/**
 * Swap a 64-bit value from bigendian to native byte order.
 *
 * If this is running on a bigendian system, `x` is returned unchanged.
 *
 * This macro never references `x` more than once, avoiding side effects.
 *
 * \param x the value to swap, in bigendian byte order.
 * \returns `x` in native byte order.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_Swap64BE(x) SwapOnlyIfNecessary(x)

/**
 * Swap a floating point value from bigendian to native byte order.
 *
 * If this is running on a bigendian system, `x` is returned unchanged.
 *
 * This macro never references `x` more than once, avoiding side effects.
 *
 * \param x the value to swap, in bigendian byte order.
 * \returns `x` in native byte order.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_SwapFloatBE(x) SwapOnlyIfNecessary(x)

#elif SDL_BYTEORDER == SDL_LIL_ENDIAN
#define SDL_Swap16LE(x)     (x)
#define SDL_Swap32LE(x)     (x)
#define SDL_Swap64LE(x)     (x)
#define SDL_SwapFloatLE(x)  (x)
#define SDL_Swap16BE(x)     SDL_Swap16(x)
#define SDL_Swap32BE(x)     SDL_Swap32(x)
#define SDL_Swap64BE(x)     SDL_Swap64(x)
#define SDL_SwapFloatBE(x)  SDL_SwapFloat(x)
#else
#define SDL_Swap16LE(x)     SDL_Swap16(x)
#define SDL_Swap32LE(x)     SDL_Swap32(x)
#define SDL_Swap64LE(x)     SDL_Swap64(x)
#define SDL_SwapFloatLE(x)  SDL_SwapFloat(x)
#define SDL_Swap16BE(x)     (x)
#define SDL_Swap32BE(x)     (x)
#define SDL_Swap64BE(x)     (x)
#define SDL_SwapFloatBE(x)  (x)
#endif

/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif
#include <SDL3/SDL_close_code.h>

#endif /* SDL_endian_h_ */
