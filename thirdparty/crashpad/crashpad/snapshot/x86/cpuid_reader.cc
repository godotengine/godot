// Copyright 2017 The Crashpad Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "snapshot/x86/cpuid_reader.h"

#include <stddef.h>

#include "build/build_config.h"
#include "snapshot/cpu_context.h"

#if defined(OS_WIN)
#include <immintrin.h>
#include <intrin.h>
#endif  // OS_WIN

namespace crashpad {
namespace internal {

CpuidReader::CpuidReader()
    : features_(0),
      extended_features_(0),
      vendor_(),
      max_leaf_(0),
      signature_(0) {
  uint32_t cpuinfo[4];
  Cpuid(cpuinfo, 0);
  max_leaf_ = cpuinfo[0];
  vendor_.append(reinterpret_cast<char*>(&cpuinfo[1]), 4);
  vendor_.append(reinterpret_cast<char*>(&cpuinfo[3]), 4);
  vendor_.append(reinterpret_cast<char*>(&cpuinfo[2]), 4);

  Cpuid(cpuinfo, 1);
  signature_ = cpuinfo[0];
  features_ = (static_cast<uint64_t>(cpuinfo[2]) << 32) |
              static_cast<uint64_t>(cpuinfo[3]);

  Cpuid(cpuinfo, 0x80000001);
  extended_features_ = (static_cast<uint64_t>(cpuinfo[2]) << 32) |
                       static_cast<uint64_t>(cpuinfo[3]);
}

CpuidReader::~CpuidReader() {}

uint32_t CpuidReader::Revision() const {
  uint8_t stepping = signature_ & 0xf;
  uint8_t model = (signature_ & 0xf0) >> 4;
  uint8_t family = (signature_ & 0xf00) >> 8;
  uint8_t extended_model = static_cast<uint8_t>((signature_ & 0xf0000) >> 16);
  uint16_t extended_family = (signature_ & 0xff00000) >> 20;

  // For families before 15, extended_family are simply reserved bits.
  if (family < 15)
    extended_family = 0;
  // extended_model is only used for families 6 and 15.
  if (family != 6 && family != 15)
    extended_model = 0;

  uint16_t adjusted_family = family + extended_family;
  uint8_t adjusted_model = model + (extended_model << 4);
  return (adjusted_family << 16) | (adjusted_model << 8) | stepping;
}

uint32_t CpuidReader::Leaf7Features() const {
  if (max_leaf_ < 7) {
    return 0;
  }
  uint32_t cpuinfo[4];
  Cpuid(cpuinfo, 7);
  return cpuinfo[1];
}

bool CpuidReader::SupportsDAZ() const {
  // The correct way to check for denormals-as-zeros (DAZ) support is to examine
  // mxcsr mask, which can be done with fxsave. See Intel Software Developer’s
  // Manual, Volume 1: Basic Architecture (253665-051), 11.6.3 “Checking for the
  // DAZ Flag in the MXCSR Register”. Note that since this function tests for
  // DAZ support in the CPU, it checks the mxcsr mask. Testing mxcsr would
  // indicate whether DAZ is actually enabled, which is a per-thread context
  // concern.

  // Test for fxsave support.
  if (!(features_ & (UINT64_C(1) << 24))) {
    return false;
  }

#if defined(ARCH_CPU_X86)
  using Fxsave = CPUContextX86::Fxsave;
#elif defined(ARCH_CPU_X86_64)
  using Fxsave = CPUContextX86_64::Fxsave;
#endif

#if defined(OS_WIN)
  __declspec(align(16)) Fxsave fxsave = {};
#else
  Fxsave fxsave __attribute__((aligned(16))) = {};
#endif

  static_assert(sizeof(fxsave) == 512, "fxsave size");
  static_assert(offsetof(decltype(fxsave), mxcsr_mask) == 28,
                "mxcsr_mask offset");

#if defined(OS_WIN)
  _fxsave(&fxsave);
#else
  asm("fxsave %0" : "=m"(fxsave));
#endif

  // Test the DAZ bit.
  return (fxsave.mxcsr_mask & (1 << 6)) != 0;
}

void CpuidReader::Cpuid(uint32_t cpuinfo[4], uint32_t leaf) const {
#if defined(OS_WIN)
  __cpuid(reinterpret_cast<int*>(cpuinfo), leaf);
#else
  asm("cpuid"
      : "=a"(cpuinfo[0]), "=b"(cpuinfo[1]), "=c"(cpuinfo[2]), "=d"(cpuinfo[3])
      : "a"(leaf), "b"(0), "c"(0), "d"(0));
#endif  // OS_WIN
}

}  // namespace internal
}  // namespace crashpad
