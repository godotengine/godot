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

#ifndef SLJIT_CONFIG_CPU_H_
#define SLJIT_CONFIG_CPU_H_

/* --------------------------------------------------------------------- */
/*  Architecture                                                         */
/* --------------------------------------------------------------------- */

/* Architecture selection. */
/* #define SLJIT_CONFIG_X86_32 1 */
/* #define SLJIT_CONFIG_X86_64 1 */
/* #define SLJIT_CONFIG_ARM_V6 1 */
/* #define SLJIT_CONFIG_ARM_V7 1 */
/* #define SLJIT_CONFIG_ARM_THUMB2 1 */
/* #define SLJIT_CONFIG_ARM_64 1 */
/* #define SLJIT_CONFIG_PPC_32 1 */
/* #define SLJIT_CONFIG_PPC_64 1 */
/* #define SLJIT_CONFIG_MIPS_32 1 */
/* #define SLJIT_CONFIG_MIPS_64 1 */
/* #define SLJIT_CONFIG_RISCV_32 1 */
/* #define SLJIT_CONFIG_RISCV_64 1 */
/* #define SLJIT_CONFIG_S390X 1 */
/* #define SLJIT_CONFIG_LOONGARCH_64 */

/* #define SLJIT_CONFIG_AUTO 1 */
/* #define SLJIT_CONFIG_UNSUPPORTED 1 */

/*****************/
/* Sanity check. */
/*****************/

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32) \
	+ (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64) \
	+ (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6) \
	+ (defined SLJIT_CONFIG_ARM_V7 && SLJIT_CONFIG_ARM_V7) \
	+ (defined SLJIT_CONFIG_ARM_THUMB2 && SLJIT_CONFIG_ARM_THUMB2) \
	+ (defined SLJIT_CONFIG_ARM_64 && SLJIT_CONFIG_ARM_64) \
	+ (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32) \
	+ (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64) \
	+ (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32) \
	+ (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64) \
	+ (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32) \
	+ (defined SLJIT_CONFIG_RISCV_64 && SLJIT_CONFIG_RISCV_64) \
	+ (defined SLJIT_CONFIG_S390X && SLJIT_CONFIG_S390X) \
	+ (defined SLJIT_CONFIG_LOONGARCH_64 && SLJIT_CONFIG_LOONGARCH_64) \
	+ (defined SLJIT_CONFIG_AUTO && SLJIT_CONFIG_AUTO) \
	+ (defined SLJIT_CONFIG_UNSUPPORTED && SLJIT_CONFIG_UNSUPPORTED) >= 2
#error "Multiple architectures are selected"
#endif

#if !(defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32) \
	&& !(defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64) \
	&& !(defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6) \
	&& !(defined SLJIT_CONFIG_ARM_V7 && SLJIT_CONFIG_ARM_V7) \
	&& !(defined SLJIT_CONFIG_ARM_THUMB2 && SLJIT_CONFIG_ARM_THUMB2) \
	&& !(defined SLJIT_CONFIG_ARM_64 && SLJIT_CONFIG_ARM_64) \
	&& !(defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32) \
	&& !(defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64) \
	&& !(defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32) \
	&& !(defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64) \
	&& !(defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32) \
	&& !(defined SLJIT_CONFIG_RISCV_64 && SLJIT_CONFIG_RISCV_64) \
	&& !(defined SLJIT_CONFIG_S390X && SLJIT_CONFIG_S390X) \
	&& !(defined SLJIT_CONFIG_LOONGARCH_64 && SLJIT_CONFIG_LOONGARCH_64) \
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
#elif defined(__aarch64__)
#define SLJIT_CONFIG_ARM_64 1
#elif defined(__thumb2__)
#define SLJIT_CONFIG_ARM_THUMB2 1
#elif (defined(__ARM_ARCH) && __ARM_ARCH >= 7) || \
	((defined(__ARM_ARCH_7__) || defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_7R__) || defined(__ARM_ARCH_7S__)) \
	 || (defined(__ARM_ARCH_8A__) || defined(__ARM_ARCH_8R__)) \
	 || (defined(__ARM_ARCH_9A__)))
#define SLJIT_CONFIG_ARM_V7 1
#elif defined(__arm__) || defined (__ARM__)
#define SLJIT_CONFIG_ARM_V6 1
#elif defined(__ppc64__) || defined(__powerpc64__) || (defined(_ARCH_PPC64) && defined(__64BIT__)) || (defined(_POWER) && defined(__64BIT__))
#define SLJIT_CONFIG_PPC_64 1
#elif defined(__ppc__) || defined(__powerpc__) || defined(_ARCH_PPC) || defined(_ARCH_PWR) || defined(_ARCH_PWR2) || defined(_POWER)
#define SLJIT_CONFIG_PPC_32 1
#elif defined(__mips__) && !defined(_LP64)
#define SLJIT_CONFIG_MIPS_32 1
#elif defined(__mips64)
#define SLJIT_CONFIG_MIPS_64 1
#elif defined (__riscv_xlen) && (__riscv_xlen == 32)
#define SLJIT_CONFIG_RISCV_32 1
#elif defined (__riscv_xlen) && (__riscv_xlen == 64)
#define SLJIT_CONFIG_RISCV_64 1
#elif defined (__loongarch_lp64)
#define SLJIT_CONFIG_LOONGARCH_64 1
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
#define SLJIT_CONFIG_ARM_V6 1
#elif defined(_M_ARM64) || defined(__aarch64__)
#define SLJIT_CONFIG_ARM_64 1
#else
#define SLJIT_CONFIG_X86_32 1
#endif

#endif /* !_WIN32 */
#endif /* SLJIT_CONFIG_AUTO */

#if (defined SLJIT_CONFIG_UNSUPPORTED && SLJIT_CONFIG_UNSUPPORTED)
#undef SLJIT_EXECUTABLE_ALLOCATOR
#endif /* SLJIT_CONFIG_UNSUPPORTED */

/******************************/
/* CPU family type detection. */
/******************************/

#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6) || (defined SLJIT_CONFIG_ARM_V7 && SLJIT_CONFIG_ARM_V7) \
	|| (defined SLJIT_CONFIG_ARM_THUMB2 && SLJIT_CONFIG_ARM_THUMB2)
#define SLJIT_CONFIG_ARM_32 1
#endif /* SLJIT_CONFIG_ARM_V6 || SLJIT_CONFIG_ARM_V7 || SLJIT_CONFIG_ARM_THUMB2 */

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32) || (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
#define SLJIT_CONFIG_X86 1
#elif (defined SLJIT_CONFIG_ARM_32 && SLJIT_CONFIG_ARM_32) || (defined SLJIT_CONFIG_ARM_64 && SLJIT_CONFIG_ARM_64)
#define SLJIT_CONFIG_ARM 1
#elif (defined SLJIT_CONFIG_PPC_32 && SLJIT_CONFIG_PPC_32) || (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
#define SLJIT_CONFIG_PPC 1
#elif (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32) || (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
#define SLJIT_CONFIG_MIPS 1
#elif (defined SLJIT_CONFIG_RISCV_32 && SLJIT_CONFIG_RISCV_32) || (defined SLJIT_CONFIG_RISCV_64 && SLJIT_CONFIG_RISCV_64)
#define SLJIT_CONFIG_RISCV 1
#elif (defined SLJIT_CONFIG_LOONGARCH_64 && SLJIT_CONFIG_LOONGARCH_64)
#define SLJIT_CONFIG_LOONGARCH 1
#endif

#endif /* SLJIT_CONFIG_CPU_H_ */
