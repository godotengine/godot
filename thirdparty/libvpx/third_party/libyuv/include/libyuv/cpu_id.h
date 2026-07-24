/*
 *  Copyright 2011 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef INCLUDE_LIBYUV_CPU_ID_H_
#define INCLUDE_LIBYUV_CPU_ID_H_

#include "libyuv/basic_types.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// Internal flag to indicate cpuid requires initialization.
static const int kCpuInitialized = 0x1;

// These flags are only valid on ARM processors.
static const int kCpuHasARM = 0x2;
static const int kCpuHasNEON = 0x4;
// 0x8 reserved for future ARM flag.

// These flags are only valid on x86 processors.
static const int kCpuHasX86 = 0x10;
static const int kCpuHasSSE2 = 0x20;
static const int kCpuHasSSSE3 = 0x40;
static const int kCpuHasSSE41 = 0x80;
static const int kCpuHasSSE42 = 0x100;  // unused at this time.
static const int kCpuHasAVX = 0x200;
static const int kCpuHasAVX2 = 0x400;
static const int kCpuHasERMS = 0x800;
static const int kCpuHasFMA3 = 0x1000;
static const int kCpuHasF16C = 0x2000;
static const int kCpuHasGFNI = 0x4000;
static const int kCpuHasAVX512BW = 0x8000;
static const int kCpuHasAVX512VL = 0x10000;
static const int kCpuHasAVX512VBMI = 0x20000;
static const int kCpuHasAVX512VBMI2 = 0x40000;
static const int kCpuHasAVX512VBITALG = 0x80000;
static const int kCpuHasAVX512VPOPCNTDQ = 0x100000;

// These flags are only valid on MIPS processors.
static const int kCpuHasMIPS = 0x200000;
static const int kCpuHasMSA = 0x400000;

// Optional init function. TestCpuFlag does an auto-init.
// Returns cpu_info flags.
LIBYUV_API
int InitCpuFlags(void);

// Detect CPU has SSE2 etc.
// Test_flag parameter should be one of kCpuHas constants above.
// Returns non-zero if instruction set is detected
static __inline int TestCpuFlag(int test_flag) {
  LIBYUV_API extern int cpu_info_;
#ifdef __ATOMIC_RELAXED
  int cpu_info = __atomic_load_n(&cpu_info_, __ATOMIC_RELAXED);
#else
  int cpu_info = cpu_info_;
#endif
  return (!cpu_info ? InitCpuFlags() : cpu_info) & test_flag;
}

// Internal function for parsing /proc/cpuinfo.
LIBYUV_API
int ArmCpuCaps(const char* cpuinfo_name);

// For testing, allow CPU flags to be disabled.
// ie MaskCpuFlags(~kCpuHasSSSE3) to disable SSSE3.
// MaskCpuFlags(-1) to enable all cpu specific optimizations.
// MaskCpuFlags(1) to disable all cpu specific optimizations.
// MaskCpuFlags(0) to reset state so next call will auto init.
// Returns cpu_info flags.
LIBYUV_API
int MaskCpuFlags(int enable_flags);

// Sets the CPU flags to |cpu_flags|, bypassing the detection code. |cpu_flags|
// should be a valid combination of the kCpuHas constants above and include
// kCpuInitialized. Use this method when running in a sandboxed process where
// the detection code might fail (as it might access /proc/cpuinfo). In such
// cases the cpu_info can be obtained from a non sandboxed process by calling
// InitCpuFlags() and passed to the sandboxed process (via command line
// parameters, IPC...) which can then call this method to initialize the CPU
// flags.
// Notes:
// - when specifying 0 for |cpu_flags|, the auto initialization is enabled
//   again.
// - enabling CPU features that are not supported by the CPU will result in
//   undefined behavior.
// TODO(fbarchard): consider writing a helper function that translates from
// other library CPU info to libyuv CPU info and add a .md doc that explains
// CPU detection.
static __inline void SetCpuFlags(int cpu_flags) {
  LIBYUV_API extern int cpu_info_;
#ifdef __ATOMIC_RELAXED
  __atomic_store_n(&cpu_info_, cpu_flags, __ATOMIC_RELAXED);
#else
  cpu_info_ = cpu_flags;
#endif
}

// Low level cpuid for X86. Returns zeros on other CPUs.
// eax is the info type that you want.
// ecx is typically the cpu number, and should normally be zero.
LIBYUV_API
void CpuId(int info_eax, int info_ecx, int* cpu_info);

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // INCLUDE_LIBYUV_CPU_ID_H_
