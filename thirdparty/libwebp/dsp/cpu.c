// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// CPU detection
//
// Author: Christian Duvivier (cduvivier@google.com)

#include "./dsp.h"

#if defined(WEBP_HAVE_NEON_RTCD)
#include <stdio.h>
#include <string.h>
#endif

#if defined(WEBP_ANDROID_NEON)
#include <cpu-features.h>
#endif

//------------------------------------------------------------------------------
// SSE2 detection.
//

// apple/darwin gcc-4.0.1 defines __PIC__, but not __pic__ with -fPIC.
#if (defined(__pic__) || defined(__PIC__)) && defined(__i386__)
static WEBP_INLINE void GetCPUInfo(int cpu_info[4], int info_type) {
  __asm__ volatile (
    "mov %%ebx, %%edi\n"
    "cpuid\n"
    "xchg %%edi, %%ebx\n"
    : "=a"(cpu_info[0]), "=D"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
    : "a"(info_type), "c"(0));
}
#elif defined(__x86_64__) && \
      (defined(__code_model_medium__) || defined(__code_model_large__)) && \
      defined(__PIC__)
static WEBP_INLINE void GetCPUInfo(int cpu_info[4], int info_type) {
  __asm__ volatile (
    "xchg{q}\t{%%rbx}, %q1\n"
    "cpuid\n"
    "xchg{q}\t{%%rbx}, %q1\n"
    : "=a"(cpu_info[0]), "=&r"(cpu_info[1]), "=c"(cpu_info[2]),
      "=d"(cpu_info[3])
    : "a"(info_type), "c"(0));
}
#elif defined(__i386__) || defined(__x86_64__)
static WEBP_INLINE void GetCPUInfo(int cpu_info[4], int info_type) {
  __asm__ volatile (
    "cpuid\n"
    : "=a"(cpu_info[0]), "=b"(cpu_info[1]), "=c"(cpu_info[2]), "=d"(cpu_info[3])
    : "a"(info_type), "c"(0));
}
#elif (defined(_M_X64) || defined(_M_IX86)) && \
      defined(_MSC_FULL_VER) && _MSC_FULL_VER >= 150030729  // >= VS2008 SP1
#include <intrin.h>
#define GetCPUInfo(info, type) __cpuidex(info, type, 0)  // set ecx=0
#elif defined(WEBP_MSC_SSE2)
#define GetCPUInfo __cpuid
#endif

// NaCl has no support for xgetbv or the raw opcode.
#if !defined(__native_client__) && (defined(__i386__) || defined(__x86_64__))
static WEBP_INLINE uint64_t xgetbv(void) {
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
static WEBP_INLINE uint64_t xgetbv(void) {
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

#if defined(__i386__) || defined(__x86_64__) || defined(WEBP_MSC_SSE2)
static int x86CPUInfo(CPUFeature feature) {
  int max_cpuid_value;
  int cpu_info[4];

  // get the highest feature value cpuid supports
  GetCPUInfo(cpu_info, 0);
  max_cpuid_value = cpu_info[0];
  if (max_cpuid_value < 1) {
    return 0;
  }

  GetCPUInfo(cpu_info, 1);
  if (feature == kSSE2) {
    return 0 != (cpu_info[3] & 0x04000000);
  }
  if (feature == kSSE3) {
    return 0 != (cpu_info[2] & 0x00000001);
  }
  if (feature == kSSE4_1) {
    return 0 != (cpu_info[2] & 0x00080000);
  }
  if (feature == kAVX) {
    // bits 27 (OSXSAVE) & 28 (256-bit AVX)
    if ((cpu_info[2] & 0x18000000) == 0x18000000) {
      // XMM state and YMM state enabled by the OS.
      return (xgetbv() & 0x6) == 0x6;
    }
  }
  if (feature == kAVX2) {
    if (x86CPUInfo(kAVX) && max_cpuid_value >= 7) {
      GetCPUInfo(cpu_info, 7);
      return ((cpu_info[1] & 0x00000020) == 0x00000020);
    }
  }
  return 0;
}
VP8CPUInfo VP8GetCPUInfo = x86CPUInfo;
#elif defined(WEBP_ANDROID_NEON)  // NB: needs to be before generic NEON test.
static int AndroidCPUInfo(CPUFeature feature) {
  const AndroidCpuFamily cpu_family = android_getCpuFamily();
  const uint64_t cpu_features = android_getCpuFeatures();
  if (feature == kNEON) {
    return (cpu_family == ANDROID_CPU_FAMILY_ARM &&
            0 != (cpu_features & ANDROID_CPU_ARM_FEATURE_NEON));
  }
  return 0;
}
VP8CPUInfo VP8GetCPUInfo = AndroidCPUInfo;
#elif defined(WEBP_USE_NEON)
// define a dummy function to enable turning off NEON at runtime by setting
// VP8DecGetCPUInfo = NULL
static int armCPUInfo(CPUFeature feature) {
  if (feature != kNEON) return 0;
#if defined(__linux__) && defined(WEBP_HAVE_NEON_RTCD)
  {
    int has_neon = 0;
    char line[200];
    FILE* const cpuinfo = fopen("/proc/cpuinfo", "r");
    if (cpuinfo == NULL) return 0;
    while (fgets(line, sizeof(line), cpuinfo)) {
      if (!strncmp(line, "Features", 8)) {
        if (strstr(line, " neon ") != NULL) {
          has_neon = 1;
          break;
        }
      }
    }
    fclose(cpuinfo);
    return has_neon;
  }
#else
  return 1;
#endif
}
VP8CPUInfo VP8GetCPUInfo = armCPUInfo;
#elif defined(WEBP_USE_MIPS32) || defined(WEBP_USE_MIPS_DSP_R2) || \
      defined(WEBP_USE_MSA)
static int mipsCPUInfo(CPUFeature feature) {
  if ((feature == kMIPS32) || (feature == kMIPSdspR2) || (feature == kMSA)) {
    return 1;
  } else {
    return 0;
  }

}
VP8CPUInfo VP8GetCPUInfo = mipsCPUInfo;
#else
VP8CPUInfo VP8GetCPUInfo = NULL;
#endif

