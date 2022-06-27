/*
 *    Stack-less Just-In-Time compiler
 *
 *    Copyright Zoltan Herczeg (hzmester@freemail.hu). All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are
 * permitted provided that the following conditions are met:
 *
 *   1. Redistributions of source code must retain the above copyright notice, this list of
 *      conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above copyright notice, this list
 *      of conditions and the following disclaimer in the documentation and/or other materials
 *      provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER(S) AND CONTRIBUTORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDER(S) OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SLJIT_CONFIG_INTERNAL_H_
#define SLJIT_CONFIG_INTERNAL_H_

#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE) \
	|| (defined SLJIT_DEBUG && SLJIT_DEBUG && (!defined(SLJIT_ASSERT) || !defined(SLJIT_UNREACHABLE)))
#include <stdio.h>
#endif

#if (defined SLJIT_DEBUG && SLJIT_DEBUG \
	&& (!defined(SLJIT_ASSERT) || !defined(SLJIT_UNREACHABLE) || !defined(SLJIT_HALT_PROCESS)))
#include <stdlib.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
   SLJIT defines the following architecture dependent types and macros:

   Types:
     sljit_s8, sljit_u8   : signed and unsigned 8 bit integer type
     sljit_s16, sljit_u16 : signed and unsigned 16 bit integer type
     sljit_s32, sljit_u32 : signed and unsigned 32 bit integer type
     sljit_sw, sljit_uw   : signed and unsigned machine word, enough to store a pointer
     sljit_p              : unsgined pointer value (usually the same as sljit_uw, but
                            some 64 bit ABIs may use 32 bit pointers)
     sljit_f32            : 32 bit single precision floating point value
     sljit_f64            : 64 bit double precision floating point value

   Macros for feature detection (boolean):
     SLJIT_32BIT_ARCHITECTURE : 32 bit architecture
     SLJIT_64BIT_ARCHITECTURE : 64 bit architecture
     SLJIT_LITTLE_ENDIAN : little endian architecture
     SLJIT_BIG_ENDIAN : big endian architecture
     SLJIT_UNALIGNED : allows unaligned memory accesses for non-fpu operations (only!)
     SLJIT_INDIRECT_CALL : see SLJIT_FUNC_ADDR() for more information

   Constants:
     SLJIT_NUMBER_OF_REGISTERS : number of available registers
     SLJIT_NUMBER_OF_SCRATCH_REGISTERS : number of available scratch registers
     SLJIT_NUMBER_OF_SAVED_REGISTERS : number of available saved registers
     SLJIT_NUMBER_OF_FLOAT_REGISTERS : number of available floating point registers
     SLJIT_NUMBER_OF_SCRATCH_FLOAT_REGISTERS : number of available floating point scratch registers
     SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS : number of available floating point saved registers
     SLJIT_WORD_SHIFT : the shift required to apply when accessing a sljit_sw/sljit_uw array by index
     SLJIT_F32_SHIFT : the shift required to apply when accessing
                       a single precision floating point array by index
     SLJIT_F64_SHIFT : the shift required to apply when accessing
                       a double precision floating point array by index
     SLJIT_PREF_SHIFT_REG : x86 systems prefers ecx for shifting by register
                            the scratch register index of ecx is stored in this variable
     SLJIT_LOCALS_OFFSET : local space starting offset (SLJIT_SP + SLJIT_LOCALS_OFFSET)
     SLJIT_RETURN_ADDRESS_OFFSET : a return instruction always adds this offset to the return address

   Other macros:
     SLJIT_FUNC : calling convention attribute for both calling JIT from C and C calling back from JIT
     SLJIT_W(number) : defining 64 bit constants on 64 bit architectures (platform independent helper)
*/

/*****************/
/* Sanity check. */
/*****************/

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32) \
	+ (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64) \
	+ (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5) \
	+ (defined SLJIT_CONFIG_ARM_V7 && SLJIT_CONFIG_ARM_V7) \
	+ (defined SLJIT_CONFIG_ARM_THUMB2 && SLJIT_CONFIG_ARM_THUMB2) \
	+ (defined SLJIT_CONFIG_ARM_64 && SLJIT_CONFIG_ARM_64) \
	+ (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32) \
	+ (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64) \
	+ (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32) \
	+ (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64) \
	+ (defined SLJIT_CONFIG_SPARC_32 && SLJIT_CONFIG_SPARC_32) \
	+ (defined SLJIT_CONFIG_S390X && SLJIT_CONFIG_S390X) \
	+ (defined SLJIT_CONFIG_AUTO && SLJIT_CONFIG_AUTO) \
	+ (defined SLJIT_CONFIG_UNSUPPORTED && SLJIT_CONFIG_UNSUPPORTED) >= 2
#error "Multiple architectures are selected"
#endif

#if !(defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32) \
	&& !(defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64) \
	&& !(defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5) \
	&& !(defined SLJIT_CONFIG_ARM_V7 && SLJIT_CONFIG_ARM_V7) \
	&& !(defined SLJIT_CONFIG_ARM_THUMB2 && SLJIT_CONFIG_ARM_THUMB2) \
	&& !(defined SLJIT_CONFIG_ARM_64 && SLJIT_CONFIG_ARM_64) \
	&& !(defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32) \
	&& !(defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64) \
	&& !(defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32) \
	&& !(defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64) \
	&& !(defined SLJIT_CONFIG_SPARC_32 && SLJIT_CONFIG_SPARC_32) \
	&& !(defined SLJIT_CONFIG_S390X && SLJIT_CONFIG_S390X) \
	&& !(defined SLJIT_CONFIG_UNSUPPORTED && SLJIT_CONFIG_UNSUPPORTED) \
	&& !(defined SLJIT_CONFIG_AUTO && SLJIT_CONFIG_AUTO)
#if defined SLJIT_CONFIG_AUTO && !SLJIT_CONFIG_AUTO
#error "An architecture must be selected"
#else /* SLJIT_CONFIG_AUTO */
#define SLJIT_CONFIG_AUTO 1
#endif /* !SLJIT_CONFIG_AUTO */
#endif /* !SLJIT_CONFIG */

/********************************************************/
/* Automatic CPU detection (requires compiler support). */
/********************************************************/

#if (defined SLJIT_CONFIG_AUTO && SLJIT_CONFIG_AUTO)

#ifndef _WIN32

#if defined(__i386__) || defined(__i386)
#define SLJIT_CONFIG_X86_32 1
#elif defined(__x86_64__)
#define SLJIT_CONFIG_X86_64 1
#elif defined(__arm__) || defined(__ARM__)
#ifdef __thumb2__
#define SLJIT_CONFIG_ARM_THUMB2 1
#elif defined(__ARM_ARCH_7__) || defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_7R__)
#define SLJIT_CONFIG_ARM_V7 1
#else
#define SLJIT_CONFIG_ARM_V5 1
#endif
#elif defined (__aarch64__)
#define SLJIT_CONFIG_ARM_64 1
#elif defined(__ppc64__) || defined(__powerpc64__) || (defined(_ARCH_PPC64) && defined(__64BIT__)) || (defined(_POWER) && defined(__64BIT__))
#define SLJIT_CONFIG_PPC_64 1
#elif defined(__ppc__) || defined(__powerpc__) || defined(_ARCH_PPC) || defined(_ARCH_PWR) || defined(_ARCH_PWR2) || defined(_POWER)
#define SLJIT_CONFIG_PPC_32 1
#elif defined(__mips__) && !defined(_LP64)
#define SLJIT_CONFIG_MIPS_32 1
#elif defined(__mips64)
#define SLJIT_CONFIG_MIPS_64 1
#elif (defined(__sparc__) || defined(__sparc)) && !defined(_LP64)
#define SLJIT_CONFIG_SPARC_32 1
#elif defined(__s390x__)
#define SLJIT_CONFIG_S390X 1
#else
/* Unsupported architecture */
#define SLJIT_CONFIG_UNSUPPORTED 1
#endif

#else /* _WIN32 */

#if defined(_M_X64) || defined(__x86_64__)
#define SLJIT_CONFIG_X86_64 1
#elif (defined(_M_ARM) && _M_ARM >= 7 && defined(_M_ARMT)) || defined(__thumb2__)
#define SLJIT_CONFIG_ARM_THUMB2 1
#elif (defined(_M_ARM) && _M_ARM >= 7)
#define SLJIT_CONFIG_ARM_V7 1
#elif defined(_ARM_)
#define SLJIT_CONFIG_ARM_V5 1
#elif defined(_M_ARM64) || defined(__aarch64__)
#define SLJIT_CONFIG_ARM_64 1
#else
#define SLJIT_CONFIG_X86_32 1
#endif

#endif /* !_WIN32 */
#endif /* SLJIT_CONFIG_AUTO */

#if (defined SLJIT_CONFIG_UNSUPPORTED && SLJIT_CONFIG_UNSUPPORTED)
#undef SLJIT_EXECUTABLE_ALLOCATOR
#endif

/******************************/
/* CPU family type detection. */
/******************************/

#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5) || (defined SLJIT_CONFIG_ARM_V7 && SLJIT_CONFIG_ARM_V7) \
	|| (defined SLJIT_CONFIG_ARM_THUMB2 && SLJIT_CONFIG_ARM_THUMB2)
#define SLJIT_CONFIG_ARM_32 1
#endif

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32) || (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
#define SLJIT_CONFIG_X86 1
#elif (defined SLJIT_CONFIG_ARM_32 && SLJIT_CONFIG_ARM_32) || (defined SLJIT_CONFIG_ARM_64 && SLJIT_CONFIG_ARM_64)
#define SLJIT_CONFIG_ARM 1
#elif (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32) || (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
#define SLJIT_CONFIG_PPC 1
#elif (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32) || (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
#define SLJIT_CONFIG_MIPS 1
#elif (defined SLJIT_CONFIG_SPARC_32 && SLJIT_CONFIG_SPARC_32) || (defined SLJIT_CONFIG_SPARC_64 && SLJIT_CONFIG_SPARC_64)
#define SLJIT_CONFIG_SPARC 1
#endif

/***********************************************************/
/* Intel Control-flow Enforcement Technology (CET) spport. */
/***********************************************************/

#ifdef SLJIT_CONFIG_X86

#if defined(__CET__) && !(defined SLJIT_CONFIG_X86_CET && SLJIT_CONFIG_X86_CET)
#define SLJIT_CONFIG_X86_CET 1
#endif

#if (defined SLJIT_CONFIG_X86_CET && SLJIT_CONFIG_X86_CET) && defined(__GNUC__)
#include <x86intrin.h>
#endif

#endif /* SLJIT_CONFIG_X86 */

/**********************************/
/* External function definitions. */
/**********************************/

/* General macros:
   Note: SLJIT is designed to be independent from them as possible.

   In release mode (SLJIT_DEBUG is not defined) only the following
   external functions are needed:
*/

#ifndef SLJIT_MALLOC
#define SLJIT_MALLOC(size, allocator_data) malloc(size)
#endif

#ifndef SLJIT_FREE
#define SLJIT_FREE(ptr, allocator_data) free(ptr)
#endif

#ifndef SLJIT_MEMCPY
#define SLJIT_MEMCPY(dest, src, len) memcpy(dest, src, len)
#endif

#ifndef SLJIT_MEMMOVE
#define SLJIT_MEMMOVE(dest, src, len) memmove(dest, src, len)
#endif

#ifndef SLJIT_ZEROMEM
#define SLJIT_ZEROMEM(dest, len) memset(dest, 0, len)
#endif

/***************************/
/* Compiler helper macros. */
/***************************/

#if !defined(SLJIT_LIKELY) && !defined(SLJIT_UNLIKELY)

#if defined(__GNUC__) && (__GNUC__ >= 3)
#define SLJIT_LIKELY(x)		__builtin_expect((x), 1)
#define SLJIT_UNLIKELY(x)	__builtin_expect((x), 0)
#else
#define SLJIT_LIKELY(x)		(x)
#define SLJIT_UNLIKELY(x)	(x)
#endif

#endif /* !defined(SLJIT_LIKELY) && !defined(SLJIT_UNLIKELY) */

#ifndef SLJIT_INLINE
/* Inline functions. Some old compilers do not support them. */
#ifdef __SUNPRO_C
#if __SUNPRO_C < 0x560
#define SLJIT_INLINE
#else
#define SLJIT_INLINE inline
#endif /* __SUNPRO_C */
#else
#define SLJIT_INLINE __inline
#endif
#endif /* !SLJIT_INLINE */

#ifndef SLJIT_NOINLINE
/* Not inline functions. */
#if defined(__GNUC__)
#define SLJIT_NOINLINE __attribute__ ((noinline))
#else
#define SLJIT_NOINLINE
#endif
#endif /* !SLJIT_INLINE */

#ifndef SLJIT_UNUSED_ARG
/* Unused arguments. */
#define SLJIT_UNUSED_ARG(arg) (void)arg
#endif

/*********************************/
/* Type of public API functions. */
/*********************************/

#ifndef SLJIT_API_FUNC_ATTRIBUTE 
#if (defined SLJIT_CONFIG_STATIC && SLJIT_CONFIG_STATIC)
/* Static ABI functions. For all-in-one programs. */

#if defined(__GNUC__)
/* Disable unused warnings in gcc. */
#define SLJIT_API_FUNC_ATTRIBUTE static __attribute__((unused))
#else
#define SLJIT_API_FUNC_ATTRIBUTE static
#endif

#else
#define SLJIT_API_FUNC_ATTRIBUTE
#endif /* (defined SLJIT_CONFIG_STATIC && SLJIT_CONFIG_STATIC) */
#endif /* defined SLJIT_API_FUNC_ATTRIBUTE */

/****************************/
/* Instruction cache flush. */
/****************************/

/*
 * TODO:
 *
 * clang >= 15 could be safe to enable below
 * older versions are known to abort in some targets
 * https://github.com/PhilipHazel/pcre2/issues/92
 *
 * beware APPLE is known to have removed the code in iOS so
 * it will need to be excempted or result in broken builds
 */
#if (!defined SLJIT_CACHE_FLUSH && defined __has_builtin)
#if __has_builtin(__builtin___clear_cache) && !defined(__clang__)

/*
 * https://gcc.gnu.org/bugzilla//show_bug.cgi?id=91248
 * https://gcc.gnu.org/bugzilla//show_bug.cgi?id=93811
 * gcc's clear_cache builtin for power and sparc are broken
 */
#if !defined(SLJIT_CONFIG_PPC) && !defined(SLJIT_CONFIG_SPARC_32)
#define SLJIT_CACHE_FLUSH(from, to) \
	__builtin___clear_cache((char*)(from), (char*)(to))
#endif

#endif /* gcc >= 10 */
#endif /* (!defined SLJIT_CACHE_FLUSH && defined __has_builtin) */

#ifndef SLJIT_CACHE_FLUSH

#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86) \
	|| (defined SLJIT_CONFIG_S390X && SLJIT_CONFIG_S390X)

/* Not required to implement on archs with unified caches. */
#define SLJIT_CACHE_FLUSH(from, to)

#elif defined __APPLE__

/* Supported by all macs since Mac OS 10.5.
   However, it does not work on non-jailbroken iOS devices,
   although the compilation is successful. */
#include <libkern/OSCacheControl.h>
#define SLJIT_CACHE_FLUSH(from, to) \
	sys_icache_invalidate((void*)(from), (size_t)((char*)(to) - (char*)(from)))

#elif (defined SLJIT_CONFIG_PPC && SLJIT_CONFIG_PPC)

/* The __clear_cache() implementation of GCC is a dummy function on PowerPC. */
#define SLJIT_CACHE_FLUSH(from, to) \
	ppc_cache_flush((from), (to))
#define SLJIT_CACHE_FLUSH_OWN_IMPL 1

#elif (defined SLJIT_CONFIG_SPARC_32 && SLJIT_CONFIG_SPARC_32)

/* The __clear_cache() implementation of GCC is a dummy function on Sparc. */
#define SLJIT_CACHE_FLUSH(from, to) \
	sparc_cache_flush((from), (to))
#define SLJIT_CACHE_FLUSH_OWN_IMPL 1

#elif (defined(__GNUC__) && (__GNUC__ >= 5 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))) || defined(__clang__)

#define SLJIT_CACHE_FLUSH(from, to) \
	__builtin___clear_cache((char*)(from), (char*)(to))

#elif defined __ANDROID__

/* Android ARMv7 with gcc lacks __clear_cache; use cacheflush instead. */
#include <sys/cachectl.h>
#define SLJIT_CACHE_FLUSH(from, to) \
	cacheflush((long)(from), (long)(to), 0)

#elif defined _WIN32

#define SLJIT_CACHE_FLUSH(from, to) \
	FlushInstructionCache(GetCurrentProcess(), (void*)(from), (char*)(to) - (char*)(from))

#else

/* Call __ARM_NR_cacheflush on ARM-Linux or the corresponding MIPS syscall. */
#define SLJIT_CACHE_FLUSH(from, to) \
	__clear_cache((char*)(from), (char*)(to))

#endif

#endif /* !SLJIT_CACHE_FLUSH */

/******************************************************/
/*    Integer and floating point type definitions.    */
/******************************************************/

/* 8 bit byte type. */
typedef unsigned char sljit_u8;
typedef signed char sljit_s8;

/* 16 bit half-word type. */
typedef unsigned short int sljit_u16;
typedef signed short int sljit_s16;

/* 32 bit integer type. */
typedef unsigned int sljit_u32;
typedef signed int sljit_s32;

/* Machine word type. Enough for storing a pointer.
     32 bit for 32 bit machines.
     64 bit for 64 bit machines. */
#if (defined SLJIT_CONFIG_UNSUPPORTED && SLJIT_CONFIG_UNSUPPORTED)
/* Just to have something. */
#define SLJIT_WORD_SHIFT 0
typedef unsigned long int sljit_uw;
typedef long int sljit_sw;
#elif !(defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64) \
	&& !(defined SLJIT_CONFIG_ARM_64 && SLJIT_CONFIG_ARM_64) \
	&& !(defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64) \
	&& !(defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64) \
	&& !(defined SLJIT_CONFIG_S390X && SLJIT_CONFIG_S390X)
#define SLJIT_32BIT_ARCHITECTURE 1
#define SLJIT_WORD_SHIFT 2
typedef unsigned int sljit_uw;
typedef int sljit_sw;
#else
#define SLJIT_64BIT_ARCHITECTURE 1
#define SLJIT_WORD_SHIFT 3
#ifdef _WIN32
#ifdef __GNUC__
/* These types do not require windows.h */
typedef unsigned long long sljit_uw;
typedef long long sljit_sw;
#else
typedef unsigned __int64 sljit_uw;
typedef __int64 sljit_sw;
#endif
#else /* !_WIN32 */
typedef unsigned long int sljit_uw;
typedef long int sljit_sw;
#endif /* _WIN32 */
#endif

typedef sljit_uw sljit_p;

/* Floating point types. */
typedef float sljit_f32;
typedef double sljit_f64;

/* Shift for pointer sized data. */
#define SLJIT_POINTER_SHIFT SLJIT_WORD_SHIFT

/* Shift for double precision sized data. */
#define SLJIT_F32_SHIFT 2
#define SLJIT_F64_SHIFT 3

#ifndef SLJIT_W

/* Defining long constants. */
#if (defined SLJIT_CONFIG_UNSUPPORTED && SLJIT_CONFIG_UNSUPPORTED)
#define SLJIT_W(w)	(w##l)
#elif (defined SLJIT_64BIT_ARCHITECTURE && SLJIT_64BIT_ARCHITECTURE)
#ifdef _WIN64
#define SLJIT_W(w)	(w##ll)
#else /* !windows */
#define SLJIT_W(w)	(w##l)
#endif /* windows */
#else /* 32 bit */
#define SLJIT_W(w)	(w)
#endif /* unknown */

#endif /* !SLJIT_W */

/*************************/
/* Endianness detection. */
/*************************/

#if !defined(SLJIT_BIG_ENDIAN) && !defined(SLJIT_LITTLE_ENDIAN)

/* These macros are mostly useful for the applications. */
#if (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32) \
	|| (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)

#ifdef __LITTLE_ENDIAN__
#define SLJIT_LITTLE_ENDIAN 1
#else
#define SLJIT_BIG_ENDIAN 1
#endif

#elif (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32) \
	|| (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)

#ifdef __MIPSEL__
#define SLJIT_LITTLE_ENDIAN 1
#else
#define SLJIT_BIG_ENDIAN 1
#endif

#ifndef SLJIT_MIPS_REV

/* Auto detecting mips revision. */
#if (defined __mips_isa_rev) && (__mips_isa_rev >= 6)
#define SLJIT_MIPS_REV 6
#elif (defined __mips_isa_rev && __mips_isa_rev >= 1) \
	|| (defined __clang__ && defined _MIPS_ARCH_OCTEON) \
	|| (defined __clang__ && defined _MIPS_ARCH_P5600)
/* clang either forgets to define (clang-7) __mips_isa_rev at all
 * or sets it to zero (clang-8,-9) for -march=octeon (MIPS64 R2+)
 * and -march=p5600 (MIPS32 R5).
 * It also sets the __mips macro to 64 or 32 for -mipsN when N <= 5
 * (should be set to N exactly) so we cannot rely on this too.
 */
#define SLJIT_MIPS_REV 1
#endif

#endif /* !SLJIT_MIPS_REV */

#elif (defined SLJIT_CONFIG_SPARC_32 && SLJIT_CONFIG_SPARC_32) \
	|| (defined SLJIT_CONFIG_S390X && SLJIT_CONFIG_S390X)

#define SLJIT_BIG_ENDIAN 1

#else
#define SLJIT_LITTLE_ENDIAN 1
#endif

#endif /* !defined(SLJIT_BIG_ENDIAN) && !defined(SLJIT_LITTLE_ENDIAN) */

/* Sanity check. */
#if (defined SLJIT_BIG_ENDIAN && SLJIT_BIG_ENDIAN) && (defined SLJIT_LITTLE_ENDIAN && SLJIT_LITTLE_ENDIAN)
#error "Exactly one endianness must be selected"
#endif

#if !(defined SLJIT_BIG_ENDIAN && SLJIT_BIG_ENDIAN) && !(defined SLJIT_LITTLE_ENDIAN && SLJIT_LITTLE_ENDIAN)
#error "Exactly one endianness must be selected"
#endif

#ifndef SLJIT_UNALIGNED

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32) \
	|| (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64) \
	|| (defined SLJIT_CONFIG_ARM_V7 && SLJIT_CONFIG_ARM_V7) \
	|| (defined SLJIT_CONFIG_ARM_THUMB2 && SLJIT_CONFIG_ARM_THUMB2) \
	|| (defined SLJIT_CONFIG_ARM_64 && SLJIT_CONFIG_ARM_64) \
	|| (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32) \
	|| (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64) \
	|| (defined SLJIT_CONFIG_S390X && SLJIT_CONFIG_S390X)
#define SLJIT_UNALIGNED 1
#endif

#endif /* !SLJIT_UNALIGNED */

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
/* Auto detect SSE2 support using CPUID.
   On 64 bit x86 cpus, sse2 must be present. */
#define SLJIT_DETECT_SSE2 1
#endif

/*****************************************************************************************/
/* Calling convention of functions generated by SLJIT or called from the generated code. */
/*****************************************************************************************/

#ifndef SLJIT_FUNC

#if (defined SLJIT_USE_CDECL_CALLING_CONVENTION && SLJIT_USE_CDECL_CALLING_CONVENTION) \
	|| !(defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)

#define SLJIT_FUNC

#elif defined(__GNUC__) && !defined(__APPLE__)

#if __GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4)
#define SLJIT_FUNC __attribute__ ((fastcall))
#define SLJIT_X86_32_FASTCALL 1
#else
#define SLJIT_FUNC
#endif /* gcc >= 3.4 */

#elif defined(_MSC_VER)

#define SLJIT_FUNC __fastcall
#define SLJIT_X86_32_FASTCALL 1

#elif defined(__BORLANDC__)

#define SLJIT_FUNC __msfastcall
#define SLJIT_X86_32_FASTCALL 1

#else /* Unknown compiler. */

/* The cdecl calling convention is usually the x86 default. */
#define SLJIT_FUNC

#endif /* SLJIT_USE_CDECL_CALLING_CONVENTION */

#endif /* !SLJIT_FUNC */

#ifndef SLJIT_INDIRECT_CALL
#if ((defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64) && (!defined _CALL_ELF || _CALL_ELF == 1)) \
	|| ((defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32) && defined _AIX)
/* It seems certain ppc compilers use an indirect addressing for functions
   which makes things complicated. */
#define SLJIT_INDIRECT_CALL 1
#endif
#endif /* SLJIT_INDIRECT_CALL */

/* The offset which needs to be substracted from the return address to
determine the next executed instruction after return. */
#ifndef SLJIT_RETURN_ADDRESS_OFFSET
#if (defined SLJIT_CONFIG_SPARC_32 && SLJIT_CONFIG_SPARC_32)
#define SLJIT_RETURN_ADDRESS_OFFSET 8
#else
#define SLJIT_RETURN_ADDRESS_OFFSET 0
#endif
#endif /* SLJIT_RETURN_ADDRESS_OFFSET */

/***************************************************/
/* Functions of the built-in executable allocator. */
/***************************************************/

#if (defined SLJIT_EXECUTABLE_ALLOCATOR && SLJIT_EXECUTABLE_ALLOCATOR)
SLJIT_API_FUNC_ATTRIBUTE void* sljit_malloc_exec(sljit_uw size);
SLJIT_API_FUNC_ATTRIBUTE void sljit_free_exec(void* ptr);
SLJIT_API_FUNC_ATTRIBUTE void sljit_free_unused_memory_exec(void);
#define SLJIT_BUILTIN_MALLOC_EXEC(size, exec_allocator_data) sljit_malloc_exec(size)
#define SLJIT_BUILTIN_FREE_EXEC(ptr, exec_allocator_data) sljit_free_exec(ptr)

#ifndef SLJIT_MALLOC_EXEC
#define SLJIT_MALLOC_EXEC(size, exec_allocator_data) SLJIT_BUILTIN_MALLOC_EXEC((size), (exec_allocator_data))
#endif /* SLJIT_MALLOC_EXEC */

#ifndef SLJIT_FREE_EXEC
#define SLJIT_FREE_EXEC(ptr, exec_allocator_data) SLJIT_BUILTIN_FREE_EXEC((ptr), (exec_allocator_data))
#endif /* SLJIT_FREE_EXEC */

#if (defined SLJIT_PROT_EXECUTABLE_ALLOCATOR && SLJIT_PROT_EXECUTABLE_ALLOCATOR)
SLJIT_API_FUNC_ATTRIBUTE sljit_sw sljit_exec_offset(void* ptr);
#define SLJIT_EXEC_OFFSET(ptr) sljit_exec_offset(ptr)
#else
#define SLJIT_EXEC_OFFSET(ptr) 0
#endif

#endif /* SLJIT_EXECUTABLE_ALLOCATOR */

/**********************************************/
/* Registers and locals offset determination. */
/**********************************************/

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)

#define SLJIT_NUMBER_OF_REGISTERS 12
#define SLJIT_NUMBER_OF_SAVED_REGISTERS 9
#define SLJIT_NUMBER_OF_FLOAT_REGISTERS 7
#define SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS 0
#define SLJIT_LOCALS_OFFSET_BASE (compiler->locals_offset)
#define SLJIT_PREF_SHIFT_REG SLJIT_R2

#elif (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)

#define SLJIT_NUMBER_OF_REGISTERS 13
#define SLJIT_NUMBER_OF_FLOAT_REGISTERS 15
#ifndef _WIN64
#define SLJIT_NUMBER_OF_SAVED_REGISTERS 6
#define SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS 0
#define SLJIT_LOCALS_OFFSET_BASE 0
#else /* _WIN64 */
#define SLJIT_NUMBER_OF_SAVED_REGISTERS 8
#define SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS 10
#define SLJIT_LOCALS_OFFSET_BASE (4 * (sljit_s32)sizeof(sljit_sw))
#endif /* !_WIN64 */
#define SLJIT_PREF_SHIFT_REG SLJIT_R3

#elif (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5) || (defined SLJIT_CONFIG_ARM_V7 && SLJIT_CONFIG_ARM_V7)

#define SLJIT_NUMBER_OF_REGISTERS 12
#define SLJIT_NUMBER_OF_SAVED_REGISTERS 8
#define SLJIT_NUMBER_OF_FLOAT_REGISTERS 14
#define SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS 8
#define SLJIT_LOCALS_OFFSET_BASE 0

#elif (defined SLJIT_CONFIG_ARM_THUMB2 && SLJIT_CONFIG_ARM_THUMB2)

#define SLJIT_NUMBER_OF_REGISTERS 12
#define SLJIT_NUMBER_OF_SAVED_REGISTERS 8
#define SLJIT_NUMBER_OF_FLOAT_REGISTERS 14
#define SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS 8
#define SLJIT_LOCALS_OFFSET_BASE 0

#elif (defined SLJIT_CONFIG_ARM_64 && SLJIT_CONFIG_ARM_64)

#define SLJIT_NUMBER_OF_REGISTERS 26
#define SLJIT_NUMBER_OF_SAVED_REGISTERS 10
#define SLJIT_NUMBER_OF_FLOAT_REGISTERS 30
#define SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS 8
#define SLJIT_LOCALS_OFFSET_BASE (2 * (sljit_s32)sizeof(sljit_sw))

#elif (defined SLJIT_CONFIG_PPC && SLJIT_CONFIG_PPC)

#define SLJIT_NUMBER_OF_REGISTERS 23
#define SLJIT_NUMBER_OF_SAVED_REGISTERS 17
#define SLJIT_NUMBER_OF_FLOAT_REGISTERS 30
#define SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS 18
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64) || (defined _AIX)
#define SLJIT_LOCALS_OFFSET_BASE ((6 + 8) * (sljit_s32)sizeof(sljit_sw))
#elif (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)
/* Add +1 for double alignment. */
#define SLJIT_LOCALS_OFFSET_BASE ((3 + 1) * (sljit_s32)sizeof(sljit_sw))
#else
#define SLJIT_LOCALS_OFFSET_BASE (3 * (sljit_s32)sizeof(sljit_sw))
#endif /* SLJIT_CONFIG_PPC_64 || _AIX */

#elif (defined SLJIT_CONFIG_MIPS && SLJIT_CONFIG_MIPS)

#define SLJIT_NUMBER_OF_REGISTERS 21
#define SLJIT_NUMBER_OF_SAVED_REGISTERS 8
#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
#define SLJIT_LOCALS_OFFSET_BASE (4 * (sljit_s32)sizeof(sljit_sw))
#define SLJIT_NUMBER_OF_FLOAT_REGISTERS 13
#define SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS 6
#else
#define SLJIT_LOCALS_OFFSET_BASE 0
#define SLJIT_NUMBER_OF_FLOAT_REGISTERS 29
#define SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS 8
#endif

#elif (defined SLJIT_CONFIG_SPARC && SLJIT_CONFIG_SPARC)

#define SLJIT_NUMBER_OF_REGISTERS 18
#define SLJIT_NUMBER_OF_SAVED_REGISTERS 14
#define SLJIT_NUMBER_OF_FLOAT_REGISTERS 14
#define SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS 0
#if (defined SLJIT_CONFIG_SPARC_32 && SLJIT_CONFIG_SPARC_32)
/* saved registers (16), return struct pointer (1), space for 6 argument words (1),
   4th double arg (2), double alignment (1). */
#define SLJIT_LOCALS_OFFSET_BASE ((16 + 1 + 6 + 2 + 1) * (sljit_s32)sizeof(sljit_sw))
#endif

#elif (defined SLJIT_CONFIG_S390X && SLJIT_CONFIG_S390X)

/*
 * https://refspecs.linuxbase.org/ELF/zSeries/lzsabi0_zSeries.html#STACKFRAME
 *
 * 160
 *  .. FR6
 *  .. FR4
 *  .. FR2
 * 128 FR0
 * 120 R15 (used for SP)
 * 112 R14
 * 104 R13
 *  96 R12
 *  ..
 *  48 R6
 *  ..
 *  16 R2
 *   8 RESERVED
 *   0 SP
 */
#define SLJIT_S390X_DEFAULT_STACK_FRAME_SIZE 160

#define SLJIT_NUMBER_OF_REGISTERS 12
#define SLJIT_NUMBER_OF_SAVED_REGISTERS 8
#define SLJIT_NUMBER_OF_FLOAT_REGISTERS 15
#define SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS 8
#define SLJIT_LOCALS_OFFSET_BASE SLJIT_S390X_DEFAULT_STACK_FRAME_SIZE

#elif (defined SLJIT_CONFIG_UNSUPPORTED && SLJIT_CONFIG_UNSUPPORTED)

#define SLJIT_NUMBER_OF_REGISTERS 0
#define SLJIT_NUMBER_OF_SAVED_REGISTERS 0
#define SLJIT_NUMBER_OF_FLOAT_REGISTERS 0
#define SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS 0
#define SLJIT_LOCALS_OFFSET_BASE 0

#endif

#define SLJIT_LOCALS_OFFSET (SLJIT_LOCALS_OFFSET_BASE)

#define SLJIT_NUMBER_OF_SCRATCH_REGISTERS \
	(SLJIT_NUMBER_OF_REGISTERS - SLJIT_NUMBER_OF_SAVED_REGISTERS)

#define SLJIT_NUMBER_OF_SCRATCH_FLOAT_REGISTERS \
	(SLJIT_NUMBER_OF_FLOAT_REGISTERS - SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS)

/********************************/
/* CPU status flags management. */
/********************************/

#if (defined SLJIT_CONFIG_ARM && SLJIT_CONFIG_ARM) \
	|| (defined SLJIT_CONFIG_PPC && SLJIT_CONFIG_PPC) \
	|| (defined SLJIT_CONFIG_MIPS && SLJIT_CONFIG_MIPS) \
	|| (defined SLJIT_CONFIG_SPARC && SLJIT_CONFIG_SPARC) \
	|| (defined SLJIT_CONFIG_S390X && SLJIT_CONFIG_S390X)
#define SLJIT_HAS_STATUS_FLAGS_STATE 1
#endif

/*************************************/
/* Debug and verbose related macros. */
/*************************************/

#if (defined SLJIT_DEBUG && SLJIT_DEBUG)

#if !defined(SLJIT_ASSERT) || !defined(SLJIT_UNREACHABLE)

/* SLJIT_HALT_PROCESS must halt the process. */
#ifndef SLJIT_HALT_PROCESS
#define SLJIT_HALT_PROCESS() \
	abort();
#endif /* !SLJIT_HALT_PROCESS */

#endif /* !SLJIT_ASSERT || !SLJIT_UNREACHABLE */

/* Feel free to redefine these two macros. */
#ifndef SLJIT_ASSERT

#define SLJIT_ASSERT(x) \
	do { \
		if (SLJIT_UNLIKELY(!(x))) { \
			printf("Assertion failed at " __FILE__ ":%d\n", __LINE__); \
			SLJIT_HALT_PROCESS(); \
		} \
	} while (0)

#endif /* !SLJIT_ASSERT */

#ifndef SLJIT_UNREACHABLE

#define SLJIT_UNREACHABLE() \
	do { \
		printf("Should never been reached " __FILE__ ":%d\n", __LINE__); \
		SLJIT_HALT_PROCESS(); \
	} while (0)

#endif /* !SLJIT_UNREACHABLE */

#else /* (defined SLJIT_DEBUG && SLJIT_DEBUG) */

/* Forcing empty, but valid statements. */
#undef SLJIT_ASSERT
#undef SLJIT_UNREACHABLE

#define SLJIT_ASSERT(x) \
	do { } while (0)
#define SLJIT_UNREACHABLE() \
	do { } while (0)

#endif /* (defined SLJIT_DEBUG && SLJIT_DEBUG) */

#ifndef SLJIT_COMPILE_ASSERT

#define SLJIT_COMPILE_ASSERT(x, description) \
	switch(0) { case 0: case ((x) ? 1 : 0): break; }

#endif /* !SLJIT_COMPILE_ASSERT */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SLJIT_CONFIG_INTERNAL_H_ */
