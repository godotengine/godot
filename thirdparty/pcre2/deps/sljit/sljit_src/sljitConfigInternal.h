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

/*
   SLJIT defines the following architecture dependent types and macros:

   Types:
     sljit_s8, sljit_u8   : signed and unsigned 8 bit integer type
     sljit_s16, sljit_u16 : signed and unsigned 16 bit integer type
     sljit_s32, sljit_u32 : signed and unsigned 32 bit integer type
     sljit_sw, sljit_uw   : signed and unsigned machine word, enough to store a pointer
     sljit_sp, sljit_up   : signed and unsigned pointer value (usually the same as
                            sljit_uw, but some 64 bit ABIs may use 32 bit pointers)
     sljit_f32            : 32 bit single precision floating point value
     sljit_f64            : 64 bit double precision floating point value

   Macros for feature detection (boolean):
     SLJIT_32BIT_ARCHITECTURE : 32 bit architecture
     SLJIT_64BIT_ARCHITECTURE : 64 bit architecture
     SLJIT_LITTLE_ENDIAN : little endian architecture
     SLJIT_BIG_ENDIAN : big endian architecture
     SLJIT_UNALIGNED : unaligned memory accesses for non-fpu operations are supported
     SLJIT_FPU_UNALIGNED : unaligned memory accesses for fpu operations are supported
     SLJIT_MASKED_SHIFT : all word shifts are always masked
     SLJIT_MASKED_SHIFT32 : all 32 bit shifts are always masked
     SLJIT_INDIRECT_CALL : see SLJIT_FUNC_ADDR() for more information
     SLJIT_UPPER_BITS_IGNORED : 32 bit operations ignores the upper bits of source registers
     SLJIT_UPPER_BITS_ZERO_EXTENDED : 32 bit operations clears the upper bits of destination registers
     SLJIT_UPPER_BITS_SIGN_EXTENDED : 32 bit operations replicates the sign bit in the upper bits of destination registers
     SLJIT_UPPER_BITS_PRESERVED : 32 bit operations preserves the upper bits of destination registers

   Constants:
     SLJIT_NUMBER_OF_REGISTERS : number of available registers
     SLJIT_NUMBER_OF_SCRATCH_REGISTERS : number of available scratch registers
     SLJIT_NUMBER_OF_SAVED_REGISTERS : number of available saved registers
     SLJIT_NUMBER_OF_FLOAT_REGISTERS : number of available floating point registers
     SLJIT_NUMBER_OF_SCRATCH_FLOAT_REGISTERS : number of available scratch floating point registers
     SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS : number of available saved floating point registers
     SLJIT_NUMBER_OF_VECTOR_REGISTERS : number of available vector registers
     SLJIT_NUMBER_OF_SCRATCH_VECTOR_REGISTERS : number of available scratch vector registers
     SLJIT_NUMBER_OF_SAVED_VECTOR_REGISTERS : number of available saved vector registers
     SLJIT_NUMBER_OF_TEMPORARY_REGISTERS : number of available temporary registers
     SLJIT_NUMBER_OF_TEMPORARY_FLOAT_REGISTERS : number of available temporary floating point registers
     SLJIT_NUMBER_OF_TEMPORARY_VECTOR_REGISTERS : number of available temporary vector registers
     SLJIT_SEPARATE_VECTOR_REGISTERS : if this macro is defined, the vector registers do not
                                       overlap with floating point registers
     SLJIT_WORD_SHIFT : the shift required to apply when accessing a sljit_sw/sljit_uw array by index
     SLJIT_POINTER_SHIFT : the shift required to apply when accessing a sljit_sp/sljit_up array by index
     SLJIT_F32_SHIFT : the shift required to apply when accessing
                       a single precision floating point array by index
     SLJIT_F64_SHIFT : the shift required to apply when accessing
                       a double precision floating point array by index
     SLJIT_PREF_SHIFT_REG : x86 systems prefers ecx for shifting by register
                            the scratch register index of ecx is stored in this variable
     SLJIT_LOCALS_OFFSET : local space starting offset (SLJIT_SP + SLJIT_LOCALS_OFFSET)
     SLJIT_RETURN_ADDRESS_OFFSET : a return instruction always adds this offset to the return address
     SLJIT_CONV_MAX_FLOAT : result when a floating point value is converted to integer
                            and the floating point value is higher than the maximum integer value
                            (possible values: SLJIT_CONV_RESULT_MAX_INT or SLJIT_CONV_RESULT_MIN_INT)
     SLJIT_CONV_MIN_FLOAT : result when a floating point value is converted to integer
                            and the floating point value is lower than the minimum integer value
                            (possible values: SLJIT_CONV_RESULT_MAX_INT or SLJIT_CONV_RESULT_MIN_INT)
     SLJIT_CONV_NAN_FLOAT : result when a NaN floating point value is converted to integer
                            (possible values: SLJIT_CONV_RESULT_MAX_INT, SLJIT_CONV_RESULT_MIN_INT,
                            or SLJIT_CONV_RESULT_ZERO)

   Other macros:
     SLJIT_TMP_R0 .. R9 : accessing temporary registers
     SLJIT_TMP_R(i) : accessing temporary registers
     SLJIT_TMP_FR0 .. FR9 : accessing temporary floating point registers
     SLJIT_TMP_FR(i) : accessing temporary floating point registers
     SLJIT_TMP_VR0 .. VR9 : accessing temporary vector registers
     SLJIT_TMP_VR(i) : accessing temporary vector registers
     SLJIT_TMP_DEST_REG : a temporary register for results, see the rules below
     SLJIT_TMP_DEST_FREG : a temporary register for float results, see the rules below
     SLJIT_TMP_DEST_VREG : a temporary register for vector results, see the rules below
     SLJIT_TMP_OPT_REG : a temporary register which might not be defined, see the rules below
     SLJIT_FUNC : calling convention attribute for both calling JIT from C and C calling back from JIT
     SLJIT_W(number) : defining 64 bit constants on 64 bit architectures (platform independent helper)
     SLJIT_F64_SECOND(reg) : provides the register index of the second 32 bit part of a 64 bit
                             floating point register when SLJIT_HAS_F64_AS_F32_PAIR returns non-zero

   Temporary register rules (e.g. SLJIT_TMP_DEST_REG / SLJIT_TMP_OPT_REG):
     Sljit tries to emit instructions without using any temporary registers whenever it is possible.
     When a single temporary register is needed, it is always the "OPT" register. When two temporary
     registers are needed, both the "DEST" and "OPT" are used. The x86-32 does not define an "OPT"
     register, and handles all cases without an "OPT" register which requires an "OPT" register
     on other architectures.
*/

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
#define SLJIT_MALLOC(size, allocator_data) (malloc(size))
#endif

#ifndef SLJIT_FREE
#define SLJIT_FREE(ptr, allocator_data) (free(ptr))
#endif

#ifndef SLJIT_MEMCPY
#define SLJIT_MEMCPY(dest, src, len) (memcpy(dest, src, len))
#endif

#ifndef SLJIT_MEMMOVE
#define SLJIT_MEMMOVE(dest, src, len) (memmove(dest, src, len))
#endif

#ifndef SLJIT_ZEROMEM
#define SLJIT_ZEROMEM(dest, len) (memset(dest, 0, len))
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

#ifdef __APPLE__
#include <AvailabilityMacros.h>
#endif

/*
 * TODO:
 *
 * clang >= 15 could be safe to enable below
 * older versions are known to abort in some targets
 * https://github.com/PhilipHazel/pcre2/issues/92
 *
 * beware some vendors (ex: Microsoft, Apple) are known to have
 * removed the code to support this builtin even if the call for
 * __has_builtin reports it is available.
 *
 * make sure linking doesn't fail because __clear_cache() is
 * missing before changing it or add an exception so that the
 * system provided method that should be defined below is used
 * instead.
 */
#if (!defined SLJIT_CACHE_FLUSH && defined __has_builtin)
#if __has_builtin(__builtin___clear_cache) && !defined(__clang__)

/*
 * https://gcc.gnu.org/bugzilla//show_bug.cgi?id=91248
 * https://gcc.gnu.org/bugzilla//show_bug.cgi?id=93811
 * gcc's clear_cache builtin for power is broken
 */
#if !defined(SLJIT_CONFIG_PPC)
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

#elif defined(__APPLE__) && MAC_OS_X_VERSION_MIN_REQUIRED >= 1050

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

#elif defined(_WIN32)

#define SLJIT_CACHE_FLUSH(from, to) \
	FlushInstructionCache(GetCurrentProcess(), (void*)(from), (size_t)((char*)(to) - (char*)(from)))

#elif (defined(__GNUC__) && (__GNUC__ >= 5 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))) || defined(__clang__)

#define SLJIT_CACHE_FLUSH(from, to) \
	__builtin___clear_cache((char*)(from), (char*)(to))

#elif defined __ANDROID__

/* Android ARMv7 with gcc lacks __clear_cache; use cacheflush instead. */
#include <sys/cachectl.h>
#define SLJIT_CACHE_FLUSH(from, to) \
	cacheflush((long)(from), (long)(to), 0)

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
typedef unsigned int sljit_uw;
typedef int sljit_sw;
#elif !(defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64) \
	&& !(defined SLJIT_CONFIG_ARM_64 && SLJIT_CONFIG_ARM_64) \
	&& !(defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64) \
	&& !(defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64) \
	&& !(defined SLJIT_CONFIG_RISCV_64 && SLJIT_CONFIG_RISCV_64) \
	&& !(defined SLJIT_CONFIG_S390X && SLJIT_CONFIG_S390X) \
	&& !(defined SLJIT_CONFIG_LOONGARCH_64 && SLJIT_CONFIG_LOONGARCH_64)
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

typedef sljit_sw sljit_sp;
typedef sljit_uw sljit_up;

/* Floating point types. */
typedef float sljit_f32;
typedef double sljit_f64;

/* Shift for pointer sized data. */
#define SLJIT_POINTER_SHIFT SLJIT_WORD_SHIFT

/* Shift for double precision sized data. */
#define SLJIT_F32_SHIFT 2
#define SLJIT_F64_SHIFT 3

#define SLJIT_CONV_RESULT_MAX_INT 0
#define SLJIT_CONV_RESULT_MIN_INT 1
#define SLJIT_CONV_RESULT_ZERO 2

#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86)
#define SLJIT_CONV_MAX_FLOAT SLJIT_CONV_RESULT_MIN_INT
#define SLJIT_CONV_MIN_FLOAT SLJIT_CONV_RESULT_MIN_INT
#define SLJIT_CONV_NAN_FLOAT SLJIT_CONV_RESULT_MIN_INT
#elif (defined SLJIT_CONFIG_ARM && SLJIT_CONFIG_ARM)
#define SLJIT_CONV_MAX_FLOAT SLJIT_CONV_RESULT_MAX_INT
#define SLJIT_CONV_MIN_FLOAT SLJIT_CONV_RESULT_MIN_INT
#define SLJIT_CONV_NAN_FLOAT SLJIT_CONV_RESULT_ZERO
#elif (defined SLJIT_CONFIG_MIPS && SLJIT_CONFIG_MIPS)
#define SLJIT_CONV_MAX_FLOAT SLJIT_CONV_RESULT_MAX_INT
#define SLJIT_CONV_MIN_FLOAT SLJIT_CONV_RESULT_MAX_INT
#define SLJIT_CONV_NAN_FLOAT SLJIT_CONV_RESULT_MAX_INT
#elif (defined SLJIT_CONFIG_PPC && SLJIT_CONFIG_PPC)
#define SLJIT_CONV_MAX_FLOAT SLJIT_CONV_RESULT_MAX_INT
#define SLJIT_CONV_MIN_FLOAT SLJIT_CONV_RESULT_MIN_INT
#define SLJIT_CONV_NAN_FLOAT SLJIT_CONV_RESULT_MIN_INT
#elif (defined SLJIT_CONFIG_RISCV && SLJIT_CONFIG_RISCV)
#define SLJIT_CONV_MAX_FLOAT SLJIT_CONV_RESULT_MAX_INT
#define SLJIT_CONV_MIN_FLOAT SLJIT_CONV_RESULT_MIN_INT
#define SLJIT_CONV_NAN_FLOAT SLJIT_CONV_RESULT_MAX_INT
#elif (defined SLJIT_CONFIG_S390X && SLJIT_CONFIG_S390X)
#define SLJIT_CONV_MAX_FLOAT SLJIT_CONV_RESULT_MAX_INT
#define SLJIT_CONV_MIN_FLOAT SLJIT_CONV_RESULT_MIN_INT
#define SLJIT_CONV_NAN_FLOAT SLJIT_CONV_RESULT_MIN_INT
#elif (defined SLJIT_CONFIG_LOONGARCH && SLJIT_CONFIG_LOONGARCH)
#define SLJIT_CONV_MAX_FLOAT SLJIT_CONV_RESULT_MAX_INT
#define SLJIT_CONV_MIN_FLOAT SLJIT_CONV_RESULT_MIN_INT
#define SLJIT_CONV_NAN_FLOAT SLJIT_CONV_RESULT_ZERO
#else
#error "Result for float to integer conversion is not defined"
#endif

#ifndef SLJIT_W

/* Defining long constants. */
#if (defined SLJIT_64BIT_ARCHITECTURE && SLJIT_64BIT_ARCHITECTURE)
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
#if (defined SLJIT_CONFIG_PPC && SLJIT_CONFIG_PPC)

#ifdef __LITTLE_ENDIAN__
#define SLJIT_LITTLE_ENDIAN 1
#else
#define SLJIT_BIG_ENDIAN 1
#endif

#elif (defined SLJIT_CONFIG_MIPS && SLJIT_CONFIG_MIPS)

#ifdef __MIPSEL__
#define SLJIT_LITTLE_ENDIAN 1
#else
#define SLJIT_BIG_ENDIAN 1
#endif

#ifndef SLJIT_MIPS_REV

/* Auto detecting mips revision. */
#if (defined __mips_isa_rev) && (__mips_isa_rev >= 6)
#define SLJIT_MIPS_REV 6
#elif defined(__mips_isa_rev) && __mips_isa_rev >= 1
#define SLJIT_MIPS_REV __mips_isa_rev
#elif defined(__clang__) \
	&& (defined(_MIPS_ARCH_OCTEON) || defined(_MIPS_ARCH_P5600))
/* clang either forgets to define (clang-7) __mips_isa_rev at all
 * or sets it to zero (clang-8,-9) for -march=octeon (MIPS64 R2+)
 * and -march=p5600 (MIPS32 R5).
 * It also sets the __mips macro to 64 or 32 for -mipsN when N <= 5
 * (should be set to N exactly) so we cannot rely on this too.
 */
#define SLJIT_MIPS_REV 1
#endif

#endif /* !SLJIT_MIPS_REV */

#elif (defined SLJIT_CONFIG_S390X && SLJIT_CONFIG_S390X)

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

#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86) \
	|| (defined SLJIT_CONFIG_ARM_V7 && SLJIT_CONFIG_ARM_V7) \
	|| (defined SLJIT_CONFIG_ARM_THUMB2 && SLJIT_CONFIG_ARM_THUMB2) \
	|| (defined SLJIT_CONFIG_ARM_64 && SLJIT_CONFIG_ARM_64) \
	|| (defined SLJIT_CONFIG_PPC && SLJIT_CONFIG_PPC) \
	|| (defined SLJIT_CONFIG_RISCV && SLJIT_CONFIG_RISCV) \
	|| (defined SLJIT_CONFIG_S390X && SLJIT_CONFIG_S390X) \
	|| (defined SLJIT_CONFIG_LOONGARCH && SLJIT_CONFIG_LOONGARCH)
#define SLJIT_UNALIGNED 1
#endif

#endif /* !SLJIT_UNALIGNED */

#ifndef SLJIT_FPU_UNALIGNED

#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86) \
	|| (defined SLJIT_CONFIG_ARM_64 && SLJIT_CONFIG_ARM_64) \
	|| (defined SLJIT_CONFIG_PPC && SLJIT_CONFIG_PPC) \
	|| (defined SLJIT_CONFIG_RISCV && SLJIT_CONFIG_RISCV) \
	|| (defined SLJIT_CONFIG_S390X && SLJIT_CONFIG_S390X) \
	|| (defined SLJIT_CONFIG_LOONGARCH && SLJIT_CONFIG_LOONGARCH)
#define SLJIT_FPU_UNALIGNED 1
#endif

#endif /* !SLJIT_FPU_UNALIGNED */

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
/* Auto detect SSE2 support using CPUID.
   On 64 bit x86 cpus, sse2 must be present. */
#define SLJIT_DETECT_SSE2 1
#endif

/*****************************************************************************************/
/* Calling convention of functions generated by SLJIT or called from the generated code. */
/*****************************************************************************************/

#ifndef SLJIT_FUNC
#define SLJIT_FUNC
#endif /* !SLJIT_FUNC */

#ifndef SLJIT_INDIRECT_CALL
#if ((defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64) && (!defined _CALL_ELF || _CALL_ELF == 1)) \
	|| ((defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32) && defined _AIX)
/* It seems certain ppc compilers use an indirect addressing for functions
   which makes things complicated. */
#define SLJIT_INDIRECT_CALL 1
#endif
#endif /* SLJIT_INDIRECT_CALL */

/* The offset which needs to be subtracted from the return address to
determine the next executed instruction after return. */
#ifndef SLJIT_RETURN_ADDRESS_OFFSET
#define SLJIT_RETURN_ADDRESS_OFFSET 0
#endif /* SLJIT_RETURN_ADDRESS_OFFSET */

/***************************************************/
/* Functions of the built-in executable allocator. */
/***************************************************/

#if (defined SLJIT_EXECUTABLE_ALLOCATOR && SLJIT_EXECUTABLE_ALLOCATOR)
SLJIT_API_FUNC_ATTRIBUTE void* sljit_malloc_exec(sljit_uw size);
SLJIT_API_FUNC_ATTRIBUTE void sljit_free_exec(void* ptr);
/* Note: sljitLir.h also defines sljit_free_unused_memory_exec() function. */
#define SLJIT_BUILTIN_MALLOC_EXEC(size, exec_allocator_data) sljit_malloc_exec(size)
#define SLJIT_BUILTIN_FREE_EXEC(ptr, exec_allocator_data) sljit_free_exec(ptr)

#ifndef SLJIT_MALLOC_EXEC
#define SLJIT_MALLOC_EXEC(size, exec_allocator_data) SLJIT_BUILTIN_MALLOC_EXEC((size), (exec_allocator_data))
#endif /* SLJIT_MALLOC_EXEC */

#ifndef SLJIT_FREE_EXEC
#define SLJIT_FREE_EXEC(ptr, exec_allocator_data) SLJIT_BUILTIN_FREE_EXEC((ptr), (exec_allocator_data))
#endif /* SLJIT_FREE_EXEC */

#if (defined SLJIT_PROT_EXECUTABLE_ALLOCATOR && SLJIT_PROT_EXECUTABLE_ALLOCATOR)
SLJIT_API_FUNC_ATTRIBUTE sljit_sw sljit_exec_offset(void *code);
#define SLJIT_EXEC_OFFSET(code) sljit_exec_offset(code)
#endif /* SLJIT_PROT_EXECUTABLE_ALLOCATOR */

#endif /* SLJIT_EXECUTABLE_ALLOCATOR */

#ifndef SLJIT_EXEC_OFFSET
#define SLJIT_EXEC_OFFSET(ptr) 0
#endif

/**********************************************/
/* Registers and locals offset determination. */
/**********************************************/

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)

#define SLJIT_NUMBER_OF_REGISTERS 12
#define SLJIT_NUMBER_OF_SAVED_REGISTERS 7
#define SLJIT_NUMBER_OF_TEMPORARY_REGISTERS 1
#define SLJIT_NUMBER_OF_FLOAT_REGISTERS 7
#define SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS 0
#define SLJIT_NUMBER_OF_TEMPORARY_FLOAT_REGISTERS 1
#define SLJIT_TMP_DEST_REG SLJIT_TMP_R0
#define SLJIT_TMP_DEST_FREG SLJIT_TMP_FR0
#define SLJIT_LOCALS_OFFSET_BASE (8 * (sljit_s32)sizeof(sljit_sw))
#define SLJIT_PREF_SHIFT_REG SLJIT_R2
#define SLJIT_MASKED_SHIFT 1
#define SLJIT_MASKED_SHIFT32 1
#define SLJIT_UPPER_BITS_IGNORED 1
#define SLJIT_UPPER_BITS_ZERO_EXTENDED 1

#elif (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)

#define SLJIT_NUMBER_OF_REGISTERS 13
#define SLJIT_NUMBER_OF_TEMPORARY_REGISTERS 2
#define SLJIT_NUMBER_OF_FLOAT_REGISTERS 15
#define SLJIT_NUMBER_OF_TEMPORARY_FLOAT_REGISTERS 1
#ifndef _WIN64
#define SLJIT_NUMBER_OF_SAVED_REGISTERS 6
#define SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS 0
#define SLJIT_LOCALS_OFFSET_BASE 0
#else /* _WIN64 */
#define SLJIT_NUMBER_OF_SAVED_REGISTERS 8
#define SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS 10
#define SLJIT_LOCALS_OFFSET_BASE (4 * (sljit_s32)sizeof(sljit_sw))
#endif /* !_WIN64 */
#define SLJIT_TMP_DEST_REG SLJIT_TMP_R0
#define SLJIT_TMP_OPT_REG SLJIT_TMP_R1
#define SLJIT_TMP_DEST_FREG SLJIT_TMP_FR0
#define SLJIT_PREF_SHIFT_REG SLJIT_R3
#define SLJIT_MASKED_SHIFT 1
#define SLJIT_MASKED_SHIFT32 1
#define SLJIT_UPPER_BITS_IGNORED 1
#define SLJIT_UPPER_BITS_ZERO_EXTENDED 1

#elif (defined SLJIT_CONFIG_ARM_32 && SLJIT_CONFIG_ARM_32)

#define SLJIT_NUMBER_OF_REGISTERS 12
#define SLJIT_NUMBER_OF_SAVED_REGISTERS 8
#define SLJIT_NUMBER_OF_TEMPORARY_REGISTERS 2
#define SLJIT_NUMBER_OF_FLOAT_REGISTERS 14
#define SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS 8
#define SLJIT_NUMBER_OF_TEMPORARY_FLOAT_REGISTERS 2
#define SLJIT_TMP_DEST_REG SLJIT_TMP_R1
#define SLJIT_TMP_OPT_REG SLJIT_TMP_R0
#define SLJIT_TMP_DEST_FREG SLJIT_TMP_FR0
#define SLJIT_LOCALS_OFFSET_BASE 0

#elif (defined SLJIT_CONFIG_ARM_64 && SLJIT_CONFIG_ARM_64)

#define SLJIT_NUMBER_OF_REGISTERS 26
#define SLJIT_NUMBER_OF_SAVED_REGISTERS 10
#define SLJIT_NUMBER_OF_TEMPORARY_REGISTERS 3
#define SLJIT_NUMBER_OF_FLOAT_REGISTERS 30
#define SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS 8
#define SLJIT_NUMBER_OF_TEMPORARY_FLOAT_REGISTERS 2
#define SLJIT_TMP_DEST_REG SLJIT_TMP_R0
#define SLJIT_TMP_OPT_REG SLJIT_TMP_R1
#define SLJIT_TMP_DEST_FREG SLJIT_TMP_FR0
#define SLJIT_LOCALS_OFFSET_BASE (2 * (sljit_s32)sizeof(sljit_sw))
#define SLJIT_MASKED_SHIFT 1
#define SLJIT_MASKED_SHIFT32 1
#define SLJIT_UPPER_BITS_IGNORED 1
#define SLJIT_UPPER_BITS_ZERO_EXTENDED 1

#elif (defined SLJIT_CONFIG_PPC && SLJIT_CONFIG_PPC)

#define SLJIT_NUMBER_OF_REGISTERS 23
#define SLJIT_NUMBER_OF_SAVED_REGISTERS 17
#define SLJIT_NUMBER_OF_TEMPORARY_REGISTERS 3
#define SLJIT_NUMBER_OF_FLOAT_REGISTERS 30
#define SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS 18
#define SLJIT_NUMBER_OF_TEMPORARY_FLOAT_REGISTERS 2
#define SLJIT_TMP_DEST_REG SLJIT_TMP_R1
#define SLJIT_TMP_OPT_REG SLJIT_TMP_R0
#define SLJIT_TMP_DEST_FREG SLJIT_TMP_FR0
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64) || (defined _AIX)
#define SLJIT_LOCALS_OFFSET_BASE ((6 + 8) * (sljit_s32)sizeof(sljit_sw))
#elif (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32)
/* Add +1 for double alignment. */
#define SLJIT_LOCALS_OFFSET_BASE ((3 + 1) * (sljit_s32)sizeof(sljit_sw))
#else
#define SLJIT_LOCALS_OFFSET_BASE (3 * (sljit_s32)sizeof(sljit_sw))
#endif /* SLJIT_CONFIG_PPC_64 || _AIX */
#define SLJIT_UPPER_BITS_IGNORED 1

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
#define SLJIT_NUMBER_OF_TEMPORARY_REGISTERS 5
#define SLJIT_NUMBER_OF_TEMPORARY_FLOAT_REGISTERS 3
#define SLJIT_TMP_DEST_REG SLJIT_TMP_R1
#define SLJIT_TMP_OPT_REG SLJIT_TMP_R0
#define SLJIT_TMP_DEST_FREG SLJIT_TMP_FR0
#define SLJIT_MASKED_SHIFT 1
#define SLJIT_MASKED_SHIFT32 1
#define SLJIT_UPPER_BITS_SIGN_EXTENDED 1

#elif (defined SLJIT_CONFIG_RISCV && SLJIT_CONFIG_RISCV)

#define SLJIT_NUMBER_OF_REGISTERS 23
#define SLJIT_NUMBER_OF_SAVED_REGISTERS 12
#define SLJIT_NUMBER_OF_TEMPORARY_REGISTERS 5
#define SLJIT_NUMBER_OF_FLOAT_REGISTERS 30
#define SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS 12
#define SLJIT_NUMBER_OF_TEMPORARY_FLOAT_REGISTERS 2
#define SLJIT_SEPARATE_VECTOR_REGISTERS 1
#define SLJIT_NUMBER_OF_VECTOR_REGISTERS 30
#define SLJIT_NUMBER_OF_SAVED_VECTOR_REGISTERS 0
#define SLJIT_NUMBER_OF_TEMPORARY_VECTOR_REGISTERS 2
#define SLJIT_TMP_DEST_REG SLJIT_TMP_R1
#define SLJIT_TMP_OPT_REG SLJIT_TMP_R0
#define SLJIT_TMP_DEST_FREG SLJIT_TMP_FR0
#define SLJIT_TMP_DEST_VREG SLJIT_TMP_VR0
#define SLJIT_LOCALS_OFFSET_BASE 0
#define SLJIT_MASKED_SHIFT 1
#define SLJIT_MASKED_SHIFT32 1
#define SLJIT_UPPER_BITS_IGNORED 1
#define SLJIT_UPPER_BITS_SIGN_EXTENDED 1

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
#define SLJIT_NUMBER_OF_TEMPORARY_REGISTERS 3
#define SLJIT_NUMBER_OF_FLOAT_REGISTERS 15
#define SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS 8
#define SLJIT_NUMBER_OF_TEMPORARY_FLOAT_REGISTERS 1
#define SLJIT_TMP_DEST_REG SLJIT_TMP_R2
#define SLJIT_TMP_OPT_REG SLJIT_TMP_R1
#define SLJIT_TMP_DEST_FREG SLJIT_TMP_FR0
#define SLJIT_LOCALS_OFFSET_BASE SLJIT_S390X_DEFAULT_STACK_FRAME_SIZE
#define SLJIT_MASKED_SHIFT 1
#define SLJIT_UPPER_BITS_IGNORED 1
#define SLJIT_UPPER_BITS_PRESERVED 1

#elif (defined SLJIT_CONFIG_LOONGARCH && SLJIT_CONFIG_LOONGARCH)

#define SLJIT_NUMBER_OF_REGISTERS 23
#define SLJIT_NUMBER_OF_SAVED_REGISTERS 10
#define SLJIT_NUMBER_OF_TEMPORARY_REGISTERS 5
#define SLJIT_NUMBER_OF_FLOAT_REGISTERS 30
#define SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS 12
#define SLJIT_NUMBER_OF_TEMPORARY_FLOAT_REGISTERS 2
#define SLJIT_TMP_DEST_REG SLJIT_TMP_R1
#define SLJIT_TMP_OPT_REG SLJIT_TMP_R0
#define SLJIT_TMP_DEST_FREG SLJIT_TMP_FR0
#define SLJIT_LOCALS_OFFSET_BASE 0
#define SLJIT_MASKED_SHIFT 1
#define SLJIT_MASKED_SHIFT32 1
#define SLJIT_UPPER_BITS_SIGN_EXTENDED 1

#elif (defined SLJIT_CONFIG_UNSUPPORTED && SLJIT_CONFIG_UNSUPPORTED)

/* Just to have something. */
#define SLJIT_NUMBER_OF_REGISTERS 0
#define SLJIT_NUMBER_OF_SAVED_REGISTERS 0
#define SLJIT_NUMBER_OF_TEMPORARY_REGISTERS 0
#define SLJIT_NUMBER_OF_FLOAT_REGISTERS 0
#define SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS 0
#define SLJIT_NUMBER_OF_TEMPORARY_FLOAT_REGISTERS 0
#define SLJIT_TMP_DEST_REG 0
#define SLJIT_TMP_DEST_FREG 0
#define SLJIT_LOCALS_OFFSET_BASE 0

#endif

#if !(defined SLJIT_SEPARATE_VECTOR_REGISTERS && SLJIT_SEPARATE_VECTOR_REGISTERS)
#define SLJIT_NUMBER_OF_VECTOR_REGISTERS (SLJIT_NUMBER_OF_FLOAT_REGISTERS)
#define SLJIT_NUMBER_OF_SAVED_VECTOR_REGISTERS (SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS)
#define SLJIT_NUMBER_OF_TEMPORARY_VECTOR_REGISTERS (SLJIT_NUMBER_OF_TEMPORARY_FLOAT_REGISTERS)
#define SLJIT_TMP_DEST_VREG (SLJIT_TMP_DEST_FREG)
#endif /* !SLJIT_SEPARATE_VECTOR_REGISTERS */

#define SLJIT_LOCALS_OFFSET (SLJIT_LOCALS_OFFSET_BASE)

#define SLJIT_NUMBER_OF_SCRATCH_REGISTERS \
	(SLJIT_NUMBER_OF_REGISTERS - SLJIT_NUMBER_OF_SAVED_REGISTERS)

#define SLJIT_NUMBER_OF_SCRATCH_FLOAT_REGISTERS \
	(SLJIT_NUMBER_OF_FLOAT_REGISTERS - SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS)

#define SLJIT_NUMBER_OF_SCRATCH_VECTOR_REGISTERS \
	(SLJIT_NUMBER_OF_VECTOR_REGISTERS - SLJIT_NUMBER_OF_SAVED_VECTOR_REGISTERS)

#if (defined SLJIT_UPPER_BITS_ZERO_EXTENDED && SLJIT_UPPER_BITS_ZERO_EXTENDED) \
	+ (defined SLJIT_UPPER_BITS_SIGN_EXTENDED && SLJIT_UPPER_BITS_SIGN_EXTENDED) \
	+ (defined SLJIT_UPPER_BITS_PRESERVED && SLJIT_UPPER_BITS_PRESERVED) > 1
#error "Invalid upper bits defintion"
#endif

#if (defined SLJIT_UPPER_BITS_PRESERVED && SLJIT_UPPER_BITS_PRESERVED) \
	&& !(defined SLJIT_UPPER_BITS_IGNORED && SLJIT_UPPER_BITS_IGNORED)
#error "Upper bits preserved requires bits ignored"
#endif

/**********************************/
/* Temporary register management. */
/**********************************/

#define SLJIT_TMP_REGISTER_BASE (SLJIT_NUMBER_OF_REGISTERS + 2)
#define SLJIT_TMP_FREGISTER_BASE (SLJIT_NUMBER_OF_FLOAT_REGISTERS + 1)
#define SLJIT_TMP_VREGISTER_BASE (SLJIT_NUMBER_OF_VECTOR_REGISTERS + 1)

/* WARNING: Accessing temporary registers is not recommended, because they
   are also used by the JIT compiler for various computations. Using them
   might have any side effects including incorrect operations and crashes,
   so use them at your own risk. The machine registers themselves might have
   limitations, e.g. the r0 register on s390x / ppc cannot be used as
   base address for memory operations. */

/* Temporary registers */
#define SLJIT_TMP_R0		(SLJIT_TMP_REGISTER_BASE + 0)
#define SLJIT_TMP_R1		(SLJIT_TMP_REGISTER_BASE + 1)
#define SLJIT_TMP_R2		(SLJIT_TMP_REGISTER_BASE + 2)
#define SLJIT_TMP_R3		(SLJIT_TMP_REGISTER_BASE + 3)
#define SLJIT_TMP_R4		(SLJIT_TMP_REGISTER_BASE + 4)
#define SLJIT_TMP_R5		(SLJIT_TMP_REGISTER_BASE + 5)
#define SLJIT_TMP_R6		(SLJIT_TMP_REGISTER_BASE + 6)
#define SLJIT_TMP_R7		(SLJIT_TMP_REGISTER_BASE + 7)
#define SLJIT_TMP_R8		(SLJIT_TMP_REGISTER_BASE + 8)
#define SLJIT_TMP_R9		(SLJIT_TMP_REGISTER_BASE + 9)
#define SLJIT_TMP_R(i)		(SLJIT_TMP_REGISTER_BASE + (i))

#define SLJIT_TMP_FR0		(SLJIT_TMP_FREGISTER_BASE + 0)
#define SLJIT_TMP_FR1		(SLJIT_TMP_FREGISTER_BASE + 1)
#define SLJIT_TMP_FR2		(SLJIT_TMP_FREGISTER_BASE + 2)
#define SLJIT_TMP_FR3		(SLJIT_TMP_FREGISTER_BASE + 3)
#define SLJIT_TMP_FR4		(SLJIT_TMP_FREGISTER_BASE + 4)
#define SLJIT_TMP_FR5		(SLJIT_TMP_FREGISTER_BASE + 5)
#define SLJIT_TMP_FR6		(SLJIT_TMP_FREGISTER_BASE + 6)
#define SLJIT_TMP_FR7		(SLJIT_TMP_FREGISTER_BASE + 7)
#define SLJIT_TMP_FR8		(SLJIT_TMP_FREGISTER_BASE + 8)
#define SLJIT_TMP_FR9		(SLJIT_TMP_FREGISTER_BASE + 9)
#define SLJIT_TMP_FR(i)		(SLJIT_TMP_FREGISTER_BASE + (i))

#define SLJIT_TMP_VR0		(SLJIT_TMP_VREGISTER_BASE + 0)
#define SLJIT_TMP_VR1		(SLJIT_TMP_VREGISTER_BASE + 1)
#define SLJIT_TMP_VR2		(SLJIT_TMP_VREGISTER_BASE + 2)
#define SLJIT_TMP_VR3		(SLJIT_TMP_VREGISTER_BASE + 3)
#define SLJIT_TMP_VR4		(SLJIT_TMP_VREGISTER_BASE + 4)
#define SLJIT_TMP_VR5		(SLJIT_TMP_VREGISTER_BASE + 5)
#define SLJIT_TMP_VR6		(SLJIT_TMP_VREGISTER_BASE + 6)
#define SLJIT_TMP_VR7		(SLJIT_TMP_VREGISTER_BASE + 7)
#define SLJIT_TMP_VR8		(SLJIT_TMP_VREGISTER_BASE + 8)
#define SLJIT_TMP_VR9		(SLJIT_TMP_VREGISTER_BASE + 9)
#define SLJIT_TMP_VR(i)		(SLJIT_TMP_VREGISTER_BASE + (i))

/********************************/
/* CPU status flags management. */
/********************************/

#if (defined SLJIT_CONFIG_ARM && SLJIT_CONFIG_ARM) \
	|| (defined SLJIT_CONFIG_PPC && SLJIT_CONFIG_PPC) \
	|| (defined SLJIT_CONFIG_MIPS && SLJIT_CONFIG_MIPS) \
	|| (defined SLJIT_CONFIG_RISCV && SLJIT_CONFIG_RISCV) \
	|| (defined SLJIT_CONFIG_S390X && SLJIT_CONFIG_S390X) \
	|| (defined SLJIT_CONFIG_LOONGARCH && SLJIT_CONFIG_LOONGARCH)
#define SLJIT_HAS_STATUS_FLAGS_STATE 1
#endif

/***************************************/
/* Floating point register management. */
/***************************************/

#if (defined SLJIT_CONFIG_ARM_32 && SLJIT_CONFIG_ARM_32) \
	|| (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
#define SLJIT_F64_SECOND(reg) \
	((reg) + SLJIT_FS0 + SLJIT_NUMBER_OF_TEMPORARY_FLOAT_REGISTERS)
#else /* !SLJIT_CONFIG_ARM_32 && !SLJIT_CONFIG_MIPS_32 */
#define SLJIT_F64_SECOND(reg) \
	(reg)
#endif /* SLJIT_CONFIG_ARM_32 || SLJIT_CONFIG_MIPS_32 */

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

#ifndef SLJIT_FALLTHROUGH

#if defined(__cplusplus) && __cplusplus >= 202002L && \
	defined(__has_cpp_attribute)
/* Standards-compatible C++ variant. */
#if __has_cpp_attribute(fallthrough)
#define SLJIT_FALLTHROUGH [[fallthrough]];
#endif
#elif !defined(__cplusplus) && \
	  defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L && \
	  defined(__has_c_attribute)
/* Standards-compatible C variant. */
#if __has_c_attribute(fallthrough)
#define SLJIT_FALLTHROUGH [[fallthrough]];
#endif
#elif ((defined(__clang__) && __clang_major__ >= 10) || \
	   (defined(__GNUC__) && __GNUC__ >= 7)) && \
	  defined(__has_attribute)
/* Clang and GCC syntax. Rule out old versions because apparently Clang at
   least has a broken implementation of __has_attribute. */
#if __has_attribute(fallthrough)
#define SLJIT_FALLTHROUGH __attribute__((fallthrough));
#endif
#endif

#ifndef SLJIT_FALLTHROUGH
#define SLJIT_FALLTHROUGH
#endif

#endif /* !SLJIT_FALLTHROUGH */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SLJIT_CONFIG_INTERNAL_H_ */
