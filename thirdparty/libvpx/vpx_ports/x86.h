/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#ifndef VPX_PORTS_X86_H_
#define VPX_PORTS_X86_H_
#include <stdlib.h>

#if defined(_MSC_VER)
#include <intrin.h>  /* For __cpuidex, __rdtsc */
#endif

#include "vpx_config.h"
#include "vpx/vpx_integer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  VPX_CPU_UNKNOWN = -1,
  VPX_CPU_AMD,
  VPX_CPU_AMD_OLD,
  VPX_CPU_CENTAUR,
  VPX_CPU_CYRIX,
  VPX_CPU_INTEL,
  VPX_CPU_NEXGEN,
  VPX_CPU_NSC,
  VPX_CPU_RISE,
  VPX_CPU_SIS,
  VPX_CPU_TRANSMETA,
  VPX_CPU_TRANSMETA_OLD,
  VPX_CPU_UMC,
  VPX_CPU_VIA,

  VPX_CPU_LAST
}  vpx_cpu_t;

#if defined(__GNUC__) && __GNUC__ || defined(__ANDROID__)
#if ARCH_X86_64
#define cpuid(func, func2, ax, bx, cx, dx)\
  __asm__ __volatile__ (\
                        "cpuid           \n\t" \
                        : "=a" (ax), "=b" (bx), "=c" (cx), "=d" (dx) \
                        : "a" (func), "c" (func2));
#else
#define cpuid(func, func2, ax, bx, cx, dx)\
  __asm__ __volatile__ (\
                        "mov %%ebx, %%edi   \n\t" \
                        "cpuid              \n\t" \
                        "xchg %%edi, %%ebx  \n\t" \
                        : "=a" (ax), "=D" (bx), "=c" (cx), "=d" (dx) \
                        : "a" (func), "c" (func2));
#endif
#elif defined(__SUNPRO_C) || defined(__SUNPRO_CC) /* end __GNUC__ or __ANDROID__*/
#if ARCH_X86_64
#define cpuid(func, func2, ax, bx, cx, dx)\
  asm volatile (\
                "xchg %rsi, %rbx \n\t" \
                "cpuid           \n\t" \
                "movl %ebx, %edi \n\t" \
                "xchg %rsi, %rbx \n\t" \
                : "=a" (ax), "=D" (bx), "=c" (cx), "=d" (dx) \
                : "a" (func), "c" (func2));
#else
#define cpuid(func, func2, ax, bx, cx, dx)\
  asm volatile (\
                "pushl %ebx       \n\t" \
                "cpuid            \n\t" \
                "movl %ebx, %edi  \n\t" \
                "popl %ebx        \n\t" \
                : "=a" (ax), "=D" (bx), "=c" (cx), "=d" (dx) \
                : "a" (func), "c" (func2));
#endif
#else /* end __SUNPRO__ */
#if ARCH_X86_64
#if defined(_MSC_VER) && _MSC_VER > 1500
#define cpuid(func, func2, a, b, c, d) do {\
    int regs[4];\
    __cpuidex(regs, func, func2); \
    a = regs[0];  b = regs[1];  c = regs[2];  d = regs[3];\
  } while(0)
#else
#define cpuid(func, func2, a, b, c, d) do {\
    int regs[4];\
    __cpuid(regs, func); \
    a = regs[0];  b = regs[1];  c = regs[2];  d = regs[3];\
  } while (0)
#endif
#else
#define cpuid(func, func2, a, b, c, d)\
  __asm mov eax, func\
  __asm mov ecx, func2\
  __asm cpuid\
  __asm mov a, eax\
  __asm mov b, ebx\
  __asm mov c, ecx\
  __asm mov d, edx
#endif
#endif /* end others */

// NaCl has no support for xgetbv or the raw opcode.
#if !defined(__native_client__) && (defined(__i386__) || defined(__x86_64__))
static INLINE uint64_t xgetbv(void) {
  const uint32_t ecx = 0;
  uint32_t eax, edx;
  // Use the raw opcode for xgetbv for compatibility with older toolchains.
  __asm__ volatile (
    ".byte 0x0f, 0x01, 0xd0\n"
    : "=a"(eax), "=d"(edx) : "c" (ecx));
  return ((uint64_t)edx << 32) | eax;
}
#elif (defined(_M_X64) || defined(_M_IX86)) && \
      defined(_MSC_FULL_VER) && _MSC_FULL_VER >= 160040219  // >= VS2010 SP1
#include <immintrin.h>
#define xgetbv() _xgetbv(0)
#elif defined(_MSC_VER) && defined(_M_IX86)
static INLINE uint64_t xgetbv(void) {
  uint32_t eax_, edx_;
  __asm {
    xor ecx, ecx  // ecx = 0
    // Use the raw opcode for xgetbv for compatibility with older toolchains.
    __asm _emit 0x0f __asm _emit 0x01 __asm _emit 0xd0
    mov eax_, eax
    mov edx_, edx
  }
  return ((uint64_t)edx_ << 32) | eax_;
}
#else
#define xgetbv() 0U  // no AVX for older x64 or unrecognized toolchains.
#endif

#if defined(_MSC_VER) && _MSC_VER >= 1700
#include <windows.h>
#if WINAPI_FAMILY_PARTITION(WINAPI_FAMILY_APP)
#define getenv(x) NULL
#endif
#endif

#define HAS_MMX     0x01
#define HAS_SSE     0x02
#define HAS_SSE2    0x04
#define HAS_SSE3    0x08
#define HAS_SSSE3   0x10
#define HAS_SSE4_1  0x20
#define HAS_AVX     0x40
#define HAS_AVX2    0x80
#ifndef BIT
#define BIT(n) (1<<n)
#endif

static INLINE int
x86_simd_caps(void) {
  unsigned int flags = 0;
  unsigned int mask = ~0;
  unsigned int max_cpuid_val, reg_eax, reg_ebx, reg_ecx, reg_edx;
  char *env;
  (void)reg_ebx;

  /* See if the CPU capabilities are being overridden by the environment */
  env = getenv("VPX_SIMD_CAPS");

  if (env && *env)
    return (int)strtol(env, NULL, 0);

  env = getenv("VPX_SIMD_CAPS_MASK");

  if (env && *env)
    mask = (unsigned int)strtoul(env, NULL, 0);

  /* Ensure that the CPUID instruction supports extended features */
  cpuid(0, 0, max_cpuid_val, reg_ebx, reg_ecx, reg_edx);

  if (max_cpuid_val < 1)
    return 0;

  /* Get the standard feature flags */
  cpuid(1, 0, reg_eax, reg_ebx, reg_ecx, reg_edx);

  if (reg_edx & BIT(23)) flags |= HAS_MMX;

  if (reg_edx & BIT(25)) flags |= HAS_SSE; /* aka xmm */

  if (reg_edx & BIT(26)) flags |= HAS_SSE2; /* aka wmt */

  if (reg_ecx & BIT(0)) flags |= HAS_SSE3;

  if (reg_ecx & BIT(9)) flags |= HAS_SSSE3;

  if (reg_ecx & BIT(19)) flags |= HAS_SSE4_1;

  // bits 27 (OSXSAVE) & 28 (256-bit AVX)
  if ((reg_ecx & (BIT(27) | BIT(28))) == (BIT(27) | BIT(28))) {
    if ((xgetbv() & 0x6) == 0x6) {
      flags |= HAS_AVX;

      if (max_cpuid_val >= 7) {
        /* Get the leaf 7 feature flags. Needed to check for AVX2 support */
        cpuid(7, 0, reg_eax, reg_ebx, reg_ecx, reg_edx);

        if (reg_ebx & BIT(5)) flags |= HAS_AVX2;
      }
    }
  }

  return flags & mask;
}

// Note:
//  32-bit CPU cycle counter is light-weighted for most function performance
//  measurement. For large function (CPU time > a couple of seconds), 64-bit
//  counter should be used.
// 32-bit CPU cycle counter
static INLINE unsigned int
x86_readtsc(void) {
#if defined(__GNUC__) && __GNUC__
  unsigned int tsc;
  __asm__ __volatile__("rdtsc\n\t":"=a"(tsc):);
  return tsc;
#elif defined(__SUNPRO_C) || defined(__SUNPRO_CC)
  unsigned int tsc;
  asm volatile("rdtsc\n\t":"=a"(tsc):);
  return tsc;
#else
#if ARCH_X86_64
  return (unsigned int)__rdtsc();
#else
  __asm  rdtsc;
#endif
#endif
}
// 64-bit CPU cycle counter
static INLINE uint64_t
x86_readtsc64(void) {
#if defined(__GNUC__) && __GNUC__
  uint32_t hi, lo;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((uint64_t)hi << 32) | lo;
#elif defined(__SUNPRO_C) || defined(__SUNPRO_CC)
  uint_t hi, lo;
  asm volatile("rdtsc\n\t" : "=a"(lo), "=d"(hi));
  return ((uint64_t)hi << 32) | lo;
#else
#if ARCH_X86_64
  return (uint64_t)__rdtsc();
#else
  __asm  rdtsc;
#endif
#endif
}

#if defined(__GNUC__) && __GNUC__
#define x86_pause_hint()\
  __asm__ __volatile__ ("pause \n\t")
#elif defined(__SUNPRO_C) || defined(__SUNPRO_CC)
#define x86_pause_hint()\
  asm volatile ("pause \n\t")
#else
#if ARCH_X86_64
#define x86_pause_hint()\
  _mm_pause();
#else
#define x86_pause_hint()\
  __asm pause
#endif
#endif

#if defined(__GNUC__) && __GNUC__
static void
x87_set_control_word(unsigned short mode) {
  __asm__ __volatile__("fldcw %0" : : "m"(*&mode));
}
static unsigned short
x87_get_control_word(void) {
  unsigned short mode;
  __asm__ __volatile__("fstcw %0\n\t":"=m"(*&mode):);
    return mode;
}
#elif defined(__SUNPRO_C) || defined(__SUNPRO_CC)
static void
x87_set_control_word(unsigned short mode) {
  asm volatile("fldcw %0" : : "m"(*&mode));
}
static unsigned short
x87_get_control_word(void) {
  unsigned short mode;
  asm volatile("fstcw %0\n\t":"=m"(*&mode):);
  return mode;
}
#elif ARCH_X86_64
/* No fldcw intrinsics on Windows x64, punt to external asm */
extern void           vpx_winx64_fldcw(unsigned short mode);
extern unsigned short vpx_winx64_fstcw(void);
#define x87_set_control_word vpx_winx64_fldcw
#define x87_get_control_word vpx_winx64_fstcw
#else
static void
x87_set_control_word(unsigned short mode) {
  __asm { fldcw mode }
}
static unsigned short
x87_get_control_word(void) {
  unsigned short mode;
  __asm { fstcw mode }
  return mode;
}
#endif

static INLINE unsigned int
x87_set_double_precision(void) {
  unsigned int mode = x87_get_control_word();
  x87_set_control_word((mode&~0x300) | 0x200);
  return mode;
}


extern void vpx_reset_mmx_state(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_PORTS_X86_H_
