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

#if defined(_MSC_VER)
#include <intrin.h>  // For __cpuidex()
#endif
#if !defined(__pnacl__) && !defined(__CLR_VER) &&                           \
    !defined(__native_client__) && (defined(_M_IX86) || defined(_M_X64)) && \
    defined(_MSC_FULL_VER) && (_MSC_FULL_VER >= 160040219)
#include <immintrin.h>  // For _xgetbv()
#endif

// For ArmCpuCaps() but unittested on all platforms
#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// For functions that use the stack and have runtime checks for overflow,
// use SAFEBUFFERS to avoid additional check.
#if defined(_MSC_FULL_VER) && (_MSC_FULL_VER >= 160040219) && \
    !defined(__clang__)
#define SAFEBUFFERS __declspec(safebuffers)
#else
#define SAFEBUFFERS
#endif

// cpu_info_ variable for SIMD instruction sets detected.
LIBYUV_API int cpu_info_ = 0;

// TODO(fbarchard): Consider using int for cpuid so casting is not needed.
// Low level cpuid for X86.
#if (defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || \
     defined(__x86_64__)) &&                                     \
    !defined(__pnacl__) && !defined(__CLR_VER)
LIBYUV_API
void CpuId(int info_eax, int info_ecx, int* cpu_info) {
#if defined(_MSC_VER)
// Visual C version uses intrinsic or inline x86 assembly.
#if defined(_MSC_FULL_VER) && (_MSC_FULL_VER >= 160040219)
  __cpuidex(cpu_info, info_eax, info_ecx);
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
#else  // Visual C but not x86
  if (info_ecx == 0) {
    __cpuid(cpu_info, info_eax);
  } else {
    cpu_info[3] = cpu_info[2] = cpu_info[1] = cpu_info[0] = 0u;
  }
#endif
// GCC version uses inline x86 assembly.
#else  // defined(_MSC_VER)
  int info_ebx, info_edx;
  asm volatile(
#if defined(__i386__) && defined(__PIC__)
      // Preserve ebx for fpic 32 bit.
      "mov %%ebx, %%edi                          \n"
      "cpuid                                     \n"
      "xchg %%edi, %%ebx                         \n"
      : "=D"(info_ebx),
#else
      "cpuid                                     \n"
      : "=b"(info_ebx),
#endif  //  defined( __i386__) && defined(__PIC__)
        "+a"(info_eax), "+c"(info_ecx), "=d"(info_edx));
  cpu_info[0] = info_eax;
  cpu_info[1] = info_ebx;
  cpu_info[2] = info_ecx;
  cpu_info[3] = info_edx;
#endif  // defined(_MSC_VER)
}
#else  // (defined(_M_IX86) || defined(_M_X64) ...
LIBYUV_API
void CpuId(int eax, int ecx, int* cpu_info) {
  (void)eax;
  (void)ecx;
  cpu_info[0] = cpu_info[1] = cpu_info[2] = cpu_info[3] = 0;
}
#endif

// For VS2010 and earlier emit can be used:
//   _asm _emit 0x0f _asm _emit 0x01 _asm _emit 0xd0  // For VS2010 and earlier.
//  __asm {
//    xor        ecx, ecx    // xcr 0
//    xgetbv
//    mov        xcr0, eax
//  }
// For VS2013 and earlier 32 bit, the _xgetbv(0) optimizer produces bad code.
// https://code.google.com/p/libyuv/issues/detail?id=529
#if defined(_M_IX86) && (_MSC_VER < 1900)
#pragma optimize("g", off)
#endif
#if (defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || \
     defined(__x86_64__)) &&                                     \
    !defined(__pnacl__) && !defined(__CLR_VER) && !defined(__native_client__)
// X86 CPUs have xgetbv to detect OS saves high parts of ymm registers.
int GetXCR0() {
  int xcr0 = 0;
#if defined(_MSC_FULL_VER) && (_MSC_FULL_VER >= 160040219)
  xcr0 = (int)_xgetbv(0);  // VS2010 SP1 required.  NOLINT
#elif defined(__i386__) || defined(__x86_64__)
  asm(".byte 0x0f, 0x01, 0xd0" : "=a"(xcr0) : "c"(0) : "%edx");
#endif  // defined(__i386__) || defined(__x86_64__)
  return xcr0;
}
#else
// xgetbv unavailable to query for OSSave support.  Return 0.
#define GetXCR0() 0
#endif  // defined(_M_IX86) || defined(_M_X64) ..
// Return optimization to previous setting.
#if defined(_M_IX86) && (_MSC_VER < 1900)
#pragma optimize("g", on)
#endif

// based on libvpx arm_cpudetect.c
// For Arm, but public to allow testing on any CPU
LIBYUV_API SAFEBUFFERS int ArmCpuCaps(const char* cpuinfo_name) {
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
      // aarch64 uses asimd for Neon.
      p = strstr(cpuinfo_line, " asimd");
      if (p) {
        fclose(f);
        return kCpuHasNEON;
      }
    }
  }
  fclose(f);
  return 0;
}

// TODO(fbarchard): Consider read_msa_ir().
// TODO(fbarchard): Add unittest.
LIBYUV_API SAFEBUFFERS int MipsCpuCaps(const char* cpuinfo_name,
                                       const char ase[]) {
  char cpuinfo_line[512];
  FILE* f = fopen(cpuinfo_name, "r");
  if (!f) {
    // ase enabled if /proc/cpuinfo is unavailable.
    if (strcmp(ase, " msa") == 0) {
      return kCpuHasMSA;
    }
    return 0;
  }
  while (fgets(cpuinfo_line, sizeof(cpuinfo_line) - 1, f)) {
    if (memcmp(cpuinfo_line, "ASEs implemented", 16) == 0) {
      char* p = strstr(cpuinfo_line, ase);
      if (p) {
        fclose(f);
        if (strcmp(ase, " msa") == 0) {
          return kCpuHasMSA;
        }
        return 0;
      }
    }
  }
  fclose(f);
  return 0;
}

static SAFEBUFFERS int GetCpuFlags(void) {
  int cpu_info = 0;
#if !defined(__pnacl__) && !defined(__CLR_VER) &&                   \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
     defined(_M_IX86))
  int cpu_info0[4] = {0, 0, 0, 0};
  int cpu_info1[4] = {0, 0, 0, 0};
  int cpu_info7[4] = {0, 0, 0, 0};
  CpuId(0, 0, cpu_info0);
  CpuId(1, 0, cpu_info1);
  if (cpu_info0[0] >= 7) {
    CpuId(7, 0, cpu_info7);
  }
  cpu_info = kCpuHasX86 | ((cpu_info1[3] & 0x04000000) ? kCpuHasSSE2 : 0) |
             ((cpu_info1[2] & 0x00000200) ? kCpuHasSSSE3 : 0) |
             ((cpu_info1[2] & 0x00080000) ? kCpuHasSSE41 : 0) |
             ((cpu_info1[2] & 0x00100000) ? kCpuHasSSE42 : 0) |
             ((cpu_info7[1] & 0x00000200) ? kCpuHasERMS : 0);

  // AVX requires OS saves YMM registers.
  if (((cpu_info1[2] & 0x1c000000) == 0x1c000000) &&  // AVX and OSXSave
      ((GetXCR0() & 6) == 6)) {  // Test OS saves YMM registers
    cpu_info |= kCpuHasAVX | ((cpu_info7[1] & 0x00000020) ? kCpuHasAVX2 : 0) |
                ((cpu_info1[2] & 0x00001000) ? kCpuHasFMA3 : 0) |
                ((cpu_info1[2] & 0x20000000) ? kCpuHasF16C : 0);

    // Detect AVX512bw
    if ((GetXCR0() & 0xe0) == 0xe0) {
      cpu_info |= (cpu_info7[1] & 0x40000000) ? kCpuHasAVX512BW : 0;
      cpu_info |= (cpu_info7[1] & 0x80000000) ? kCpuHasAVX512VL : 0;
      cpu_info |= (cpu_info7[2] & 0x00000002) ? kCpuHasAVX512VBMI : 0;
      cpu_info |= (cpu_info7[2] & 0x00000040) ? kCpuHasAVX512VBMI2 : 0;
      cpu_info |= (cpu_info7[2] & 0x00001000) ? kCpuHasAVX512VBITALG : 0;
      cpu_info |= (cpu_info7[2] & 0x00004000) ? kCpuHasAVX512VPOPCNTDQ : 0;
      cpu_info |= (cpu_info7[2] & 0x00000100) ? kCpuHasGFNI : 0;
    }
  }
#endif
#if defined(__mips__) && defined(__linux__)
#if defined(__mips_msa)
  cpu_info = MipsCpuCaps("/proc/cpuinfo", " msa");
#endif
  cpu_info |= kCpuHasMIPS;
#endif
#if defined(__arm__) || defined(__aarch64__)
// gcc -mfpu=neon defines __ARM_NEON__
// __ARM_NEON__ generates code that requires Neon.  NaCL also requires Neon.
// For Linux, /proc/cpuinfo can be tested but without that assume Neon.
#if defined(__ARM_NEON__) || defined(__native_client__) || !defined(__linux__)
  cpu_info = kCpuHasNEON;
// For aarch64(arm64), /proc/cpuinfo's feature is not complete, e.g. no neon
// flag in it.
// So for aarch64, neon enabling is hard coded here.
#endif
#if defined(__aarch64__)
  cpu_info = kCpuHasNEON;
#else
  // Linux arm parse text file for neon detect.
  cpu_info = ArmCpuCaps("/proc/cpuinfo");
#endif
  cpu_info |= kCpuHasARM;
#endif  // __arm__
  cpu_info |= kCpuInitialized;
  return cpu_info;
}

// Note that use of this function is not thread safe.
LIBYUV_API
int MaskCpuFlags(int enable_flags) {
  int cpu_info = GetCpuFlags() & enable_flags;
  SetCpuFlags(cpu_info);
  return cpu_info;
}

LIBYUV_API
int InitCpuFlags(void) {
  return MaskCpuFlags(-1);
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
