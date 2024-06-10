// Copyright 2012 Google LLC
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google LLC nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include "common/mac/arch_utilities.h"

#include <mach/machine.h>
#include <mach-o/arch.h>
#include <mach-o/fat.h>
#include <stdio.h>
#include <string.h>

#ifdef __APPLE__
#include <Availability.h>
#include <AvailabilityMacros.h>

#if (defined(__IPHONE_OS_VERSION_MIN_REQUIRED) && defined(__IPHONE_16_0) &&    \
     __IPHONE_OS_VERSION_MIN_REQUIRED >= __IPHONE_16_0) ||                     \
    (defined(MAC_OS_X_VERSION_MIN_REQUIRED) && defined(MAC_OS_VERSION_13_0) && \
     MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_VERSION_13_0) ||                  \
    (defined(__TV_OS_VERSION_MIN_REQUIRED) && defined(__TV_OS_VERSION_16_0) && \
     __TV_OS_VERSION_MIN_REQUIRED >= __TVOS_16_0)
#define HAS_MACHO_UTILS 1
#include <mach-o/utils.h>
#else
#define HAS_MACHO_UTILS 0
#endif
#endif

namespace {

enum Architecture {
  kArch_i386 = 0,
  kArch_x86_64,
  kArch_x86_64h,
  kArch_arm,
  kArch_arm64,
  kArch_arm64e,
  kArch_ppc,
  // This must be last.
  kNumArchitectures
};

struct NamedArchInfo {
  const char* name;
  ArchInfo info;
};

// enum Architecture above and kKnownArchitectures below
// must be kept in sync.
constexpr NamedArchInfo kKnownArchitectures[] = {
    {"i386", {CPU_TYPE_I386, CPU_SUBTYPE_I386_ALL}},
    {"x86_64", {CPU_TYPE_X86_64, CPU_SUBTYPE_X86_64_ALL}},
    {"x86_64h", {CPU_TYPE_X86_64, CPU_SUBTYPE_X86_64_H}},
    {"arm", {CPU_TYPE_ARM, CPU_SUBTYPE_ARM_ALL}},
    {"arm64", {CPU_TYPE_ARM64, CPU_SUBTYPE_ARM64_ALL}},
    {"arm64e", {CPU_TYPE_ARM64, CPU_SUBTYPE_ARM64E}},
    {"ppc", {CPU_TYPE_POWERPC, CPU_SUBTYPE_POWERPC_ALL}}};

}  // namespace

ArchInfo GetLocalArchInfo(void) {
  Architecture arch;
#if defined(__i386__)
  arch = kArch_i386;
#elif defined(__x86_64__)
  arch = kArch_x86_64;
#elif defined(__arm64__) || defined(__aarch64__)
  arch = kArch_arm64;
#elif defined(__arm__)
  arch = kArch_arm;
#elif defined(__powerpc__)
  arch = kArch_ppc;
#else
  #error "Unsupported CPU architecture"
#endif
  return kKnownArchitectures[arch].info;
}

#ifdef __APPLE__

std::optional<ArchInfo> GetArchInfoFromName(const char* arch_name) {
#if HAS_MACHO_UTILS
  if (__builtin_available(macOS 13.0, iOS 16.0, tvOS 16.0, *)) {
    cpu_type_t type;
    cpu_subtype_t subtype;
    if (macho_cpu_type_for_arch_name(arch_name, &type, &subtype)) {
      return ArchInfo{type, subtype};
    }
    return std::nullopt;
  }
#endif
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  const NXArchInfo* info = NXGetArchInfoFromName(arch_name);
#pragma clang diagnostic pop
  if (info) {
    return ArchInfo{info->cputype, info->cpusubtype};
  }
  return std::nullopt;
}

const char* GetNameFromCPUType(cpu_type_t cpu_type, cpu_subtype_t cpu_subtype) {
#if HAS_MACHO_UTILS
  if (__builtin_available(macOS 13.0, iOS 16.0, tvOS 16.0, *)) {
    const char* name = macho_arch_name_for_cpu_type(cpu_type, cpu_subtype);
    if (name) {
      return name;
    }
    return kUnknownArchName;
  }
#endif
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  const NXArchInfo* info = NXGetArchInfoFromCpuType(cpu_type, cpu_subtype);
#pragma clang diagnostic pop
  if (info) {
    return info->name;
  }
  return kUnknownArchName;
}

#else

std::optional<ArchInfo> GetArchInfoFromName(const char* arch_name) {
  for (int arch = 0; arch < kNumArchitectures; ++arch) {
    if (!strcmp(arch_name, kKnownArchitectures[arch].name)) {
      return kKnownArchitectures[arch].info;
    }
  }
  return std::nullopt;
}

const char* GetNameFromCPUType(cpu_type_t cpu_type, cpu_subtype_t cpu_subtype) {
  const char* candidate = kUnknownArchName;
  for (int arch = 0; arch < kNumArchitectures; ++arch) {
    if (kKnownArchitectures[arch].info.cputype == cpu_type) {
      if (kKnownArchitectures[arch].info.cpusubtype == cpu_subtype) {
        return kKnownArchitectures[arch].name;
      }
      if (!strcmp(candidate, kUnknownArchName)) {
        candidate = kKnownArchitectures[arch].name;
      }
    }
  }
  return candidate;
}
#endif  // __APPLE__
