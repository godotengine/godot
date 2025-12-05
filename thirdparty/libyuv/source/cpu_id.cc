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
#include <stdio.h>  // For fopen()
#include <string.h>

#if defined(__linux__) && (defined(__aarch64__) || defined(__loongarch__))
#include <sys/auxv.h>  // For getauxval()
#endif

#if defined(_WIN32) && defined(__aarch64__)
#undef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#undef WIN32_EXTRA_LEAN
#define WIN32_EXTRA_LEAN
#include <windows.h>  // For IsProcessorFeaturePresent()
#endif

#if defined(__APPLE__) && defined(__aarch64__)
#include <sys/sysctl.h>  // For sysctlbyname()
#endif

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
      "mov         %%ebx, %%edi                  \n"
      "cpuid                                     \n"
      "xchg        %%edi, %%ebx                  \n"
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
#if defined(_M_IX86) && defined(_MSC_VER) && (_MSC_VER < 1900)
#pragma optimize("g", off)
#endif
#if (defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || \
     defined(__x86_64__)) &&                                     \
    !defined(__pnacl__) && !defined(__CLR_VER) && !defined(__native_client__)
// X86 CPUs have xgetbv to detect OS saves high parts of ymm registers.
static int GetXCR0() {
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
#if defined(_M_IX86) && defined(_MSC_VER) && (_MSC_VER < 1900)
#pragma optimize("g", on)
#endif

static int cpuinfo_search(const char* cpuinfo_line,
                          const char* needle,
                          int needle_len) {
  const char* p = strstr(cpuinfo_line, needle);
  return p && (p[needle_len] == ' ' || p[needle_len] == '\n');
}

// Based on libvpx arm_cpudetect.c
// For Arm, but public to allow testing on any CPU
LIBYUV_API SAFEBUFFERS int ArmCpuCaps(const char* cpuinfo_name) {
  char cpuinfo_line[512];
  FILE* f = fopen(cpuinfo_name, "re");
  if (!f) {
    // Assume Neon if /proc/cpuinfo is unavailable.
    // This will occur for Chrome sandbox for Pepper or Render process.
    return kCpuHasNEON;
  }
  memset(cpuinfo_line, 0, sizeof(cpuinfo_line));
  int features = 0;
  while (fgets(cpuinfo_line, sizeof(cpuinfo_line), f)) {
    if (memcmp(cpuinfo_line, "Features", 8) == 0) {
      if (cpuinfo_search(cpuinfo_line, " neon", 5)) {
        features |= kCpuHasNEON;
      }
    }
  }
  fclose(f);
  return features;
}

#ifdef __aarch64__
#ifdef __linux__
// Define hwcap values ourselves: building with an old auxv header where these
// hwcap values are not defined should not prevent features from being enabled.
#define YUV_AARCH64_HWCAP_ASIMDDP (1UL << 20)
#define YUV_AARCH64_HWCAP_SVE (1UL << 22)
#define YUV_AARCH64_HWCAP2_SVE2 (1UL << 1)
#define YUV_AARCH64_HWCAP2_I8MM (1UL << 13)
#define YUV_AARCH64_HWCAP2_SME (1UL << 23)
#define YUV_AARCH64_HWCAP2_SME2 (1UL << 37)

// For AArch64, but public to allow testing on any CPU.
LIBYUV_API SAFEBUFFERS int AArch64CpuCaps(unsigned long hwcap,
                                          unsigned long hwcap2) {
  // Neon is mandatory on AArch64, so enable regardless of hwcaps.
  int features = kCpuHasNEON;

  // Don't try to enable later extensions unless earlier extensions are also
  // reported available. Some of these constraints aren't strictly required by
  // the architecture, but are satisfied by all micro-architectures of
  // interest. This also avoids an issue on some emulators where true
  // architectural constraints are not satisfied, e.g. SVE2 may be reported as
  // available while SVE is not.
  if (hwcap & YUV_AARCH64_HWCAP_ASIMDDP) {
    features |= kCpuHasNeonDotProd;
    if (hwcap2 & YUV_AARCH64_HWCAP2_I8MM) {
      features |= kCpuHasNeonI8MM;
      if (hwcap & YUV_AARCH64_HWCAP_SVE) {
        features |= kCpuHasSVE;
        if (hwcap2 & YUV_AARCH64_HWCAP2_SVE2) {
          features |= kCpuHasSVE2;
        }
      }
      // SME may be present without SVE
      if (hwcap2 & YUV_AARCH64_HWCAP2_SME) {
        features |= kCpuHasSME;
        if (hwcap2 & YUV_AARCH64_HWCAP2_SME2) {
          features |= kCpuHasSME2;
        }
      }
    }
  }
  return features;
}

#elif defined(_WIN32)
// For AArch64, but public to allow testing on any CPU.
LIBYUV_API SAFEBUFFERS int AArch64CpuCaps() {
  // Neon is mandatory on AArch64, so enable unconditionally.
  int features = kCpuHasNEON;

  // For more information on IsProcessorFeaturePresent(), see:
  // https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-isprocessorfeaturepresent#parameters
#ifdef PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE
  if (IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE)) {
    features |= kCpuHasNeonDotProd;
  }
#endif
  // No Neon I8MM or SVE feature detection available here at time of writing.
  return features;
}

#elif defined(__APPLE__)
static bool have_feature(const char* feature) {
  // For more information on sysctlbyname(), see:
  // https://developer.apple.com/documentation/kernel/1387446-sysctlbyname/determining_instruction_set_characteristics
  int64_t feature_present = 0;
  size_t size = sizeof(feature_present);
  if (sysctlbyname(feature, &feature_present, &size, NULL, 0) != 0) {
    return false;
  }
  return feature_present;
}

// For AArch64, but public to allow testing on any CPU.
LIBYUV_API SAFEBUFFERS int AArch64CpuCaps() {
  // Neon is mandatory on AArch64, so enable unconditionally.
  int features = kCpuHasNEON;

  if (have_feature("hw.optional.arm.FEAT_DotProd")) {
    features |= kCpuHasNeonDotProd;
    if (have_feature("hw.optional.arm.FEAT_I8MM")) {
      features |= kCpuHasNeonI8MM;
      if (have_feature("hw.optional.arm.FEAT_SME")) {
        features |= kCpuHasSME;
        if (have_feature("hw.optional.arm.FEAT_SME2")) {
          features |= kCpuHasSME2;
        }
      }
    }
  }
  // No SVE feature detection available here at time of writing.
  return features;
}

#else  // !defined(__linux__) && !defined(_WIN32) && !defined(__APPLE__)
// For AArch64, but public to allow testing on any CPU.
LIBYUV_API SAFEBUFFERS int AArch64CpuCaps() {
  // Neon is mandatory on AArch64, so enable unconditionally.
  int features = kCpuHasNEON;

  // TODO(libyuv:980) support feature detection on other platforms.

  return features;
}
#endif
#endif  // defined(__aarch64__)

LIBYUV_API SAFEBUFFERS int RiscvCpuCaps(const char* cpuinfo_name) {
  char cpuinfo_line[512];
  int flag = 0;
  FILE* f = fopen(cpuinfo_name, "re");
  if (!f) {
#if defined(__riscv_vector)
    // Assume RVV if /proc/cpuinfo is unavailable.
    // This will occur for Chrome sandbox for Pepper or Render process.
    return kCpuHasRVV;
#else
    return 0;
#endif
  }
  memset(cpuinfo_line, 0, sizeof(cpuinfo_line));
  while (fgets(cpuinfo_line, sizeof(cpuinfo_line), f)) {
    if (memcmp(cpuinfo_line, "isa", 3) == 0) {
      // ISA string must begin with rv64{i,e,g} for a 64-bit processor.
      char* isa = strstr(cpuinfo_line, "rv64");
      if (isa) {
        size_t isa_len = strlen(isa);
        char* extensions;
        size_t extensions_len = 0;
        size_t std_isa_len;
        // Remove the new-line character at the end of string
        if (isa[isa_len - 1] == '\n') {
          isa[--isa_len] = '\0';
        }
        // 5 ISA characters
        if (isa_len < 5) {
          fclose(f);
          return 0;
        }
        // Skip {i,e,g} canonical checking.
        // Skip rvxxx
        isa += 5;
        // Find the very first occurrence of 's', 'x' or 'z'.
        // To detect multi-letter standard, non-standard, and
        // supervisor-level extensions.
        extensions = strpbrk(isa, "zxs");
        if (extensions) {
          // Multi-letter extensions are seperated by a single underscore
          // as described in RISC-V User-Level ISA V2.2.
          char* ext = strtok(extensions, "_");
          extensions_len = strlen(extensions);
          while (ext) {
            // Search for the ZVFH (Vector FP16) extension.
            if (!strcmp(ext, "zvfh")) {
              flag |= kCpuHasRVVZVFH;
            }
            ext = strtok(NULL, "_");
          }
        }
        std_isa_len = isa_len - extensions_len - 5;
        // Detect the v in the standard single-letter extensions.
        if (memchr(isa, 'v', std_isa_len)) {
          // The RVV implied the F extension.
          flag |= kCpuHasRVV;
        }
      }
    }
#if defined(__riscv_vector)
    // Assume RVV if /proc/cpuinfo is from x86 host running QEMU.
    else if ((memcmp(cpuinfo_line, "vendor_id\t: GenuineIntel", 24) == 0) ||
             (memcmp(cpuinfo_line, "vendor_id\t: AuthenticAMD", 24) == 0)) {
      fclose(f);
      return kCpuHasRVV;
    }
#endif
  }
  fclose(f);
  return flag;
}

LIBYUV_API SAFEBUFFERS int MipsCpuCaps(const char* cpuinfo_name) {
  char cpuinfo_line[512];
  int flag = 0;
  FILE* f = fopen(cpuinfo_name, "re");
  if (!f) {
    // Assume nothing if /proc/cpuinfo is unavailable.
    // This will occur for Chrome sandbox for Pepper or Render process.
    return 0;
  }
  memset(cpuinfo_line, 0, sizeof(cpuinfo_line));
  while (fgets(cpuinfo_line, sizeof(cpuinfo_line), f)) {
    if (memcmp(cpuinfo_line, "cpu model", 9) == 0) {
      // Workaround early kernel without MSA in ASEs line.
      if (strstr(cpuinfo_line, "Loongson-2K")) {
        flag |= kCpuHasMSA;
      }
    }
    if (memcmp(cpuinfo_line, "ASEs implemented", 16) == 0) {
      if (strstr(cpuinfo_line, "msa")) {
        flag |= kCpuHasMSA;
      }
      // ASEs is the last line, so we can break here.
      break;
    }
  }
  fclose(f);
  return flag;
}

#if defined(__loongarch__) && defined(__linux__)
// Define hwcap values ourselves: building with an old auxv header where these
// hwcap values are not defined should not prevent features from being enabled.
#define YUV_LOONGARCH_HWCAP_LSX (1 << 4)
#define YUV_LOONGARCH_HWCAP_LASX (1 << 5)

LIBYUV_API SAFEBUFFERS int LoongarchCpuCaps(void) {
  int flag = 0;
  unsigned long hwcap = getauxval(AT_HWCAP);

  if (hwcap & YUV_LOONGARCH_HWCAP_LSX)
    flag |= kCpuHasLSX;

  if (hwcap & YUV_LOONGARCH_HWCAP_LASX)
    flag |= kCpuHasLASX;
  return flag;
}
#endif

static SAFEBUFFERS int GetCpuFlags(void) {
  int cpu_info = 0;
#if !defined(__pnacl__) && !defined(__CLR_VER) &&                   \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
     defined(_M_IX86))
  int cpu_info0[4] = {0, 0, 0, 0};
  int cpu_info1[4] = {0, 0, 0, 0};
  int cpu_info7[4] = {0, 0, 0, 0};
  int cpu_einfo7[4] = {0, 0, 0, 0};
  int cpu_info24[4] = {0, 0, 0, 0};
  int cpu_amdinfo21[4] = {0, 0, 0, 0};
  CpuId(0, 0, cpu_info0);
  CpuId(1, 0, cpu_info1);
  if (cpu_info0[0] >= 7) {
    CpuId(7, 0, cpu_info7);
    CpuId(7, 1, cpu_einfo7);
    CpuId(0x80000021, 0, cpu_amdinfo21);
  }
  if (cpu_info0[0] >= 0x24) {
    CpuId(0x24, 0, cpu_info24);
  }
  cpu_info = kCpuHasX86 | ((cpu_info1[3] & 0x04000000) ? kCpuHasSSE2 : 0) |
             ((cpu_info1[2] & 0x00000200) ? kCpuHasSSSE3 : 0) |
             ((cpu_info1[2] & 0x00080000) ? kCpuHasSSE41 : 0) |
             ((cpu_info1[2] & 0x00100000) ? kCpuHasSSE42 : 0) |
             ((cpu_info7[1] & 0x00000200) ? kCpuHasERMS : 0) |
             ((cpu_info7[3] & 0x00000010) ? kCpuHasFSMR : 0);

  // AVX requires OS saves YMM registers.
  if (((cpu_info1[2] & 0x1c000000) == 0x1c000000) &&  // AVX and OSXSave
      ((GetXCR0() & 6) == 6)) {  // Test OS saves YMM registers
    cpu_info |= kCpuHasAVX | ((cpu_info7[1] & 0x00000020) ? kCpuHasAVX2 : 0) |
                ((cpu_info1[2] & 0x00001000) ? kCpuHasFMA3 : 0) |
                ((cpu_info1[2] & 0x20000000) ? kCpuHasF16C : 0) |
                ((cpu_einfo7[0] & 0x00000010) ? kCpuHasAVXVNNI : 0) |
                ((cpu_einfo7[3] & 0x00000010) ? kCpuHasAVXVNNIINT8 : 0);

    cpu_info |= ((cpu_amdinfo21[0] & 0x00008000) ? kCpuHasERMS : 0);

    // Detect AVX512bw
    if ((GetXCR0() & 0xe0) == 0xe0) {
      cpu_info |= ((cpu_info7[1] & 0x40000000) ? kCpuHasAVX512BW : 0) |
                  ((cpu_info7[1] & 0x80000000) ? kCpuHasAVX512VL : 0) |
                  ((cpu_info7[2] & 0x00000002) ? kCpuHasAVX512VBMI : 0) |
                  ((cpu_info7[2] & 0x00000040) ? kCpuHasAVX512VBMI2 : 0) |
                  ((cpu_info7[2] & 0x00000800) ? kCpuHasAVX512VNNI : 0) |
                  ((cpu_info7[2] & 0x00001000) ? kCpuHasAVX512VBITALG : 0) |
                  ((cpu_einfo7[3] & 0x00080000) ? kCpuHasAVX10 : 0) |
                  ((cpu_info7[3] & 0x02000000) ? kCpuHasAMXINT8 : 0);
      if (cpu_info0[0] >= 0x24 && (cpu_einfo7[3] & 0x00080000)) {
        cpu_info |= ((cpu_info24[1] & 0xFF) >= 2) ? kCpuHasAVX10_2 : 0;
      }
    }
  }
#endif
#if defined(__mips__) && defined(__linux__)
  cpu_info = MipsCpuCaps("/proc/cpuinfo");
  cpu_info |= kCpuHasMIPS;
#endif
#if defined(__loongarch__) && defined(__linux__)
  cpu_info = LoongarchCpuCaps();
  cpu_info |= kCpuHasLOONGARCH;
#endif
#if defined(__aarch64__)
#if defined(__linux__)
  // getauxval is supported since Android SDK version 18, minimum at time of
  // writing is 21, so should be safe to always use this. If getauxval is
  // somehow disabled then getauxval returns 0, which will leave Neon enabled
  // since Neon is mandatory on AArch64.
  unsigned long hwcap = getauxval(AT_HWCAP);
  unsigned long hwcap2 = getauxval(AT_HWCAP2);
  cpu_info = AArch64CpuCaps(hwcap, hwcap2);
#else
  cpu_info = AArch64CpuCaps();
#endif
  cpu_info |= kCpuHasARM;
#endif  // __aarch64__
#if defined(__arm__)
  // gcc -mfpu=neon defines __ARM_NEON__
  // __ARM_NEON__ generates code that requires Neon.  NaCL also requires Neon.
  // For Linux, /proc/cpuinfo can be tested but without that assume Neon.
  // Linux arm parse text file for neon detect.
#if defined(__linux__)
  cpu_info = ArmCpuCaps("/proc/cpuinfo");
#elif defined(__ARM_NEON__)
  cpu_info = kCpuHasNEON;
#else
  cpu_info = 0;
#endif
  cpu_info |= kCpuHasARM;
#endif  // __arm__
#if defined(__riscv) && defined(__linux__)
  cpu_info = RiscvCpuCaps("/proc/cpuinfo");
  cpu_info |= kCpuHasRISCV;
#endif  // __riscv
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
