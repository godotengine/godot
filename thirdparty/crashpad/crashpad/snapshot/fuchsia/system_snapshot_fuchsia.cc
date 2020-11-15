// Copyright 2018 The Crashpad Authors. All rights reserved.
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

#include "snapshot/fuchsia/system_snapshot_fuchsia.h"

#include <zircon/syscalls.h>

#include "base/fuchsia/fuchsia_logging.h"
#include "base/logging.h"
#include "base/numerics/safe_conversions.h"
#include "base/strings/stringprintf.h"
#include "snapshot/posix/timezone.h"

namespace crashpad {
namespace internal {

SystemSnapshotFuchsia::SystemSnapshotFuchsia() = default;

SystemSnapshotFuchsia::~SystemSnapshotFuchsia() = default;

void SystemSnapshotFuchsia::Initialize(const timeval* snapshot_time) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  snapshot_time_ = snapshot_time;

  // This version string mirrors `uname -a` as written by
  // garnet/bin/uname/uname.c, however, this information isn't provided by
  // uname(). Additionally, uname() seems to hang if the network is in a bad
  // state when attempting to retrieve the nodename, so avoid it for now.
  char kernel_version[256] = {};
  zx_status_t status =
      zx_system_get_version(kernel_version, sizeof(kernel_version));
  ZX_LOG_IF(ERROR, status != ZX_OK, status) << "zx_system_get_version";

#if defined(ARCH_CPU_X86_64)
  static constexpr const char kArch[] = "x86_64";
#elif defined(ARCH_CPU_ARM64)
  static constexpr const char kArch[] = "aarch64";
#else
  static constexpr const char kArch[] = "unknown";
#endif
  os_version_full_ =
      base::StringPrintf("Zircon prerelease %s %s", kernel_version, kArch);

  INITIALIZATION_STATE_SET_VALID(initialized_);
}

CPUArchitecture SystemSnapshotFuchsia::GetCPUArchitecture() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

#if defined(ARCH_CPU_X86_64)
  return kCPUArchitectureX86_64;
#elif defined(ARCH_CPU_ARM64)
  return kCPUArchitectureARM64;
#else
#error Port
#endif
}

uint32_t SystemSnapshotFuchsia::CPURevision() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
#if defined(ARCH_CPU_X86_64)
  return cpuid_.Revision();
#else
  NOTREACHED();  // TODO(scottmg): https://crashpad.chromium.org/bug/196.
  return 0;
#endif
}

uint8_t SystemSnapshotFuchsia::CPUCount() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return base::saturated_cast<uint8_t>(zx_system_get_num_cpus());
}

std::string SystemSnapshotFuchsia::CPUVendor() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
#if defined(ARCH_CPU_X86_64)
  return cpuid_.Vendor();
#else
  NOTREACHED();  // TODO(scottmg): https://crashpad.chromium.org/bug/196.
  return std::string();
#endif
}

void SystemSnapshotFuchsia::CPUFrequency(uint64_t* current_hz,
                                         uint64_t* max_hz) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  // TODO(scottmg): https://crashpad.chromium.org/bug/196.
  *current_hz = 0;
  *max_hz = 0;
}

uint32_t SystemSnapshotFuchsia::CPUX86Signature() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
#if defined(ARCH_CPU_X86_64)
  return cpuid_.Signature();
#else
  NOTREACHED();  // TODO(scottmg): https://crashpad.chromium.org/bug/196.
  return 0;
#endif
}

uint64_t SystemSnapshotFuchsia::CPUX86Features() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
#if defined(ARCH_CPU_X86_64)
  return cpuid_.Features();
#else
  NOTREACHED();  // TODO(scottmg): https://crashpad.chromium.org/bug/196.
  return 0;
#endif
}

uint64_t SystemSnapshotFuchsia::CPUX86ExtendedFeatures() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
#if defined(ARCH_CPU_X86_64)
  return cpuid_.ExtendedFeatures();
#else
  NOTREACHED();  // TODO(scottmg): https://crashpad.chromium.org/bug/196.
  return 0;
#endif
}

uint32_t SystemSnapshotFuchsia::CPUX86Leaf7Features() const {
#if defined(ARCH_CPU_X86_64)
  return cpuid_.Leaf7Features();
#else
  NOTREACHED();  // TODO(scottmg): https://crashpad.chromium.org/bug/196.
  return 0;
#endif
}

bool SystemSnapshotFuchsia::CPUX86SupportsDAZ() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
#if defined(ARCH_CPU_X86_64)
  return cpuid_.SupportsDAZ();
#else
  NOTREACHED();  // TODO(scottmg): https://crashpad.chromium.org/bug/196.
  return false;
#endif
}

SystemSnapshot::OperatingSystem SystemSnapshotFuchsia::GetOperatingSystem()
    const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return kOperatingSystemFuchsia;
}

bool SystemSnapshotFuchsia::OSServer() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return false;
}

void SystemSnapshotFuchsia::OSVersion(int* major,
                                      int* minor,
                                      int* bugfix,
                                      std::string* build) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  // TODO(scottmg): https://crashpad.chromium.org/bug/196. There's no version
  // available to be reported yet.
  *major = 0;
  *minor = 0;
  *bugfix = 0;
  *build = std::string();
}

std::string SystemSnapshotFuchsia::OSVersionFull() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return os_version_full_;
}

std::string SystemSnapshotFuchsia::MachineDescription() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  // TODO(scottmg): https://crashpad.chromium.org/bug/196. Not yet available,
  // upstream ZX-1775.
  return std::string();
}

bool SystemSnapshotFuchsia::NXEnabled() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
#if defined(ARCH_CPU_X86_64)
  return cpuid_.NXEnabled();
#else
  NOTREACHED();  // TODO(scottmg): https://crashpad.chromium.org/bug/196.
  return false;
#endif
}

void SystemSnapshotFuchsia::TimeZone(DaylightSavingTimeStatus* dst_status,
                                     int* standard_offset_seconds,
                                     int* daylight_offset_seconds,
                                     std::string* standard_name,
                                     std::string* daylight_name) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  internal::TimeZone(*snapshot_time_,
                     dst_status,
                     standard_offset_seconds,
                     daylight_offset_seconds,
                     standard_name,
                     daylight_name);
}

}  // namespace internal
}  // namespace crashpad
