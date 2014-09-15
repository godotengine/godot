/*
 *  Copyright 2011 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/cpu_id.h"

#ifdef _ANDROID //libtheoraplayer addition for cpu feature detection
#include "cpu-features.h"
#endif

#ifdef _MSC_VER
#include <intrin.h>  // For __cpuidex()
#endif
#if !defined(__pnacl__) && !defined(__CLR_VER) && \
    !defined(__native_client__) && defined(_M_X64) && \
    defined(_MSC_VER) && (_MSC_FULL_VER >= 160040219)
#include <immintrin.h>  // For _xgetbv()
#endif

#if !defined(__native_client__)
#include <stdlib.h>  // For getenv()
#endif

// For ArmCpuCaps() but unittested on all platforms
#include <stdio.h>
#include <string.h>

#include "libyuv/basic_types.h"  // For CPU_X86

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// For functions that use the stack and have runtime checks for overflow,
// use SAFEBUFFERS to avoid additional check.
#if defined(_MSC_VER) && (_MSC_FULL_VER >= 160040219)
#define SAFEBUFFERS __declspec(safebuffers)
#else
#define SAFEBUFFERS
#endif

// Low level cpuid for X86. Returns zeros on other CPUs.
#if !defined(__pnacl__) && !defined(__CLR_VER) && \
    (defined(_M_IX86) || defined(_M_X64) || \
    defined(__i386__) || defined(__x86_64__))
LIBYUV_API
void CpuId(uint32 info_eax, uint32 info_ecx, uint32* cpu_info) {
#if defined(_MSC_VER)
#if (_MSC_FULL_VER >= 160040219)
  __cpuidex((int*)(cpu_info), info_eax, info_ecx);
#elif defined(_M_IX86)
  __asm {
    mov        eax, info_eax
    mov        ecx, info_ecx
    mov        edi, cpu_info
    cpuid
    mov        [edi], eax
    mov        [edi + 4], ebx
    mov        [edi + 8], ecx
    mov        [edi + 12], edx
  }
#else
  if (info_ecx == 0) {
    __cpuid((int*)(cpu_info), info_eax);
  } else {
    cpu_info[3] = cpu_info[2] = cpu_info[1] = cpu_info[0] = 0;
  }
#endif
#else  // defined(_MSC_VER)
  uint32 info_ebx, info_edx;
  asm volatile (  // NOLINT
#if defined( __i386__) && defined(__PIC__)
    // Preserve ebx for fpic 32 bit.
    "mov %%ebx, %%edi                          \n"
    "cpuid                                     \n"
    "xchg %%edi, %%ebx                         \n"
    : "=D" (info_ebx),
#else
    "cpuid                                     \n"
    : "=b" (info_ebx),
#endif  //  defined( __i386__) && defined(__PIC__)
      "+a" (info_eax), "+c" (info_ecx), "=d" (info_edx));
  cpu_info[0] = info_eax;
  cpu_info[1] = info_ebx;
  cpu_info[2] = info_ecx;
  cpu_info[3] = info_edx;
#endif  // defined(_MSC_VER)
}

#if !defined(__native_client__)
#define HAS_XGETBV
// X86 CPUs have xgetbv to detect OS saves high parts of ymm registers.
int TestOsSaveYmm() {
  uint32 xcr0 = 0u;
#if defined(_MSC_VER) && (_MSC_FULL_VER >= 160040219)
  xcr0 = (uint32)(_xgetbv(0));  // VS2010 SP1 required.
#elif defined(_M_IX86)
  __asm {
    xor        ecx, ecx    // xcr 0
    _asm _emit 0x0f _asm _emit 0x01 _asm _emit 0xd0  // For VS2010 and earlier.
    mov        xcr0, eax
  }
#elif defined(__i386__) || defined(__x86_64__)
  asm(".byte 0x0f, 0x01, 0xd0" : "=a" (xcr0) : "c" (0) : "%edx");
#endif  // defined(_MSC_VER)
  return((xcr0 & 6) == 6);  // Is ymm saved?
}
#endif  // !defined(__native_client__)
#else
LIBYUV_API
void CpuId(uint32 eax, uint32 ecx, uint32* cpu_info) {
  cpu_info[0] = cpu_info[1] = cpu_info[2] = cpu_info[3] = 0;
}
#endif

// based on libvpx arm_cpudetect.c
// For Arm, but public to allow testing on any CPU
LIBYUV_API SAFEBUFFERS
int ArmCpuCaps(const char* cpuinfo_name) {
  char cpuinfo_line[512];
  FILE* f = fopen(cpuinfo_name, "r");
  if (!f) {
    // Assume Neon if /proc/cpuinfo is unavailable.
    // This will occur for Chrome sandbox for Pepper or Render process.
    return kCpuHasNEON;
  }
  while (fgets(cpuinfo_line, sizeof(cpuinfo_line) - 1, f)) {
    if (memcmp(cpuinfo_line, "Features", 8) == 0) {
      char* p = strstr(cpuinfo_line, " neon");
      if (p && (p[5] == ' ' || p[5] == '\n')) {
        fclose(f);
        return kCpuHasNEON;
      }
    }
  }
  fclose(f);
  return 0;
}

#if defined(__mips__) && defined(__linux__)
static int MipsCpuCaps(const char* search_string) {
  char cpuinfo_line[512];
  const char* file_name = "/proc/cpuinfo";
  FILE* f = fopen(file_name, "r");
  if (!f) {
    // Assume DSP if /proc/cpuinfo is unavailable.
    // This will occur for Chrome sandbox for Pepper or Render process.
    return kCpuHasMIPS_DSP;
  }
  while (fgets(cpuinfo_line, sizeof(cpuinfo_line) - 1, f) != NULL) {
    if (strstr(cpuinfo_line, search_string) != NULL) {
      fclose(f);
      return kCpuHasMIPS_DSP;
    }
  }
  fclose(f);
  return 0;
}
#endif

// CPU detect function for SIMD instruction sets.
LIBYUV_API
int cpu_info_ = kCpuInit;  // cpu_info is not initialized yet.

// Test environment variable for disabling CPU features. Any non-zero value
// to disable. Zero ignored to make it easy to set the variable on/off.
#if !defined(__native_client__) && !defined(_M_ARM)

static LIBYUV_BOOL TestEnv(const char* name) {
#ifndef _WINRT
  const char* var = getenv(name);
  if (var) {
    if (var[0] != '0') {
      return LIBYUV_TRUE;
    }
  }
#endif
  return LIBYUV_FALSE;
}
#else  // nacl does not support getenv().
static LIBYUV_BOOL TestEnv(const char*) {
  return LIBYUV_FALSE;
}
#endif

LIBYUV_API SAFEBUFFERS
int InitCpuFlags(void) {
#if !defined(__pnacl__) && !defined(__CLR_VER) && defined(CPU_X86)

  uint32 cpu_info1[4] = { 0, 0, 0, 0 };
  uint32 cpu_info7[4] = { 0, 0, 0, 0 };
  CpuId(1, 0, cpu_info1);
  CpuId(7, 0, cpu_info7);
  cpu_info_ = ((cpu_info1[3] & 0x04000000) ? kCpuHasSSE2 : 0) |
              ((cpu_info1[2] & 0x00000200) ? kCpuHasSSSE3 : 0) |
              ((cpu_info1[2] & 0x00080000) ? kCpuHasSSE41 : 0) |
              ((cpu_info1[2] & 0x00100000) ? kCpuHasSSE42 : 0) |
              ((cpu_info7[1] & 0x00000200) ? kCpuHasERMS : 0) |
              ((cpu_info1[2] & 0x00001000) ? kCpuHasFMA3 : 0) |
              kCpuHasX86;
#ifdef HAS_XGETBV
  if ((cpu_info1[2] & 0x18000000) == 0x18000000 &&  // AVX and OSSave
      TestOsSaveYmm()) {  // Saves YMM.
    cpu_info_ |= ((cpu_info7[1] & 0x00000020) ? kCpuHasAVX2 : 0) |
                 kCpuHasAVX;
  }
#endif
  // Environment variable overrides for testing.
  if (TestEnv("LIBYUV_DISABLE_X86")) {
    cpu_info_ &= ~kCpuHasX86;
  }
  if (TestEnv("LIBYUV_DISABLE_SSE2")) {
    cpu_info_ &= ~kCpuHasSSE2;
  }
  if (TestEnv("LIBYUV_DISABLE_SSSE3")) {
    cpu_info_ &= ~kCpuHasSSSE3;
  }
  if (TestEnv("LIBYUV_DISABLE_SSE41")) {
    cpu_info_ &= ~kCpuHasSSE41;
  }
  if (TestEnv("LIBYUV_DISABLE_SSE42")) {
    cpu_info_ &= ~kCpuHasSSE42;
  }
  if (TestEnv("LIBYUV_DISABLE_AVX")) {
    cpu_info_ &= ~kCpuHasAVX;
  }
  if (TestEnv("LIBYUV_DISABLE_AVX2")) {
    cpu_info_ &= ~kCpuHasAVX2;
  }
  if (TestEnv("LIBYUV_DISABLE_ERMS")) {
    cpu_info_ &= ~kCpuHasERMS;
  }
  if (TestEnv("LIBYUV_DISABLE_FMA3")) {
    cpu_info_ &= ~kCpuHasFMA3;
  }
#elif defined(__mips__) && defined(__linux__)
  // Linux mips parse text file for dsp detect.
  cpu_info_ = MipsCpuCaps("dsp");  // set kCpuHasMIPS_DSP.
#if defined(__mips_dspr2)
  cpu_info_ |= kCpuHasMIPS_DSPR2;
#endif
  cpu_info_ |= kCpuHasMIPS;

  if (getenv("LIBYUV_DISABLE_MIPS")) {
    cpu_info_ &= ~kCpuHasMIPS;
  }
  if (getenv("LIBYUV_DISABLE_MIPS_DSP")) {
    cpu_info_ &= ~kCpuHasMIPS_DSP;
  }
  if (getenv("LIBYUV_DISABLE_MIPS_DSPR2")) {
    cpu_info_ &= ~kCpuHasMIPS_DSPR2;
  }
#elif defined(__arm__)
// gcc -mfpu=neon defines __ARM_NEON__
// __ARM_NEON__ generates code that requires Neon.  NaCL also requires Neon.
// For Linux, /proc/cpuinfo can be tested but without that assume Neon.
#if defined(__ARM_NEON__) || defined(__native_client__) || !defined(__linux__)
#ifdef _ANDROID
  cpu_info_ = ArmCpuCaps("/proc/cpuinfo"); // libtheoraplayer #ifdef addition, just in case, android gave us troubles
#else
  cpu_info_ = kCpuHasNEON;
#endif
#else
  // Linux arm parse text file for neon detect.
  cpu_info_ = ArmCpuCaps("/proc/cpuinfo");
#endif
  cpu_info_ |= kCpuHasARM;
  if (TestEnv("LIBYUV_DISABLE_NEON")) {
    cpu_info_ &= ~kCpuHasNEON;
  }
#ifdef _ANDROID
  // libtheoraplayer addition to disable NEON support on android devices that don't support it, once again, just in case	
  if ((android_getCpuFeaturesExt() & ANDROID_CPU_ARM_FEATURE_NEON) == 0)
  {
 	cpu_info_ = kCpuHasARM;
  }
#endif
#endif  // __arm__
  if (TestEnv("LIBYUV_DISABLE_ASM")) {
    cpu_info_ = 0;
  }
  return cpu_info_;
}

LIBYUV_API
void MaskCpuFlags(int enable_flags) {
  cpu_info_ = InitCpuFlags() & enable_flags;
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
