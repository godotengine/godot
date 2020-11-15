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

#ifndef CRASHPAD_SNAPSHOT_FUCHSIA_SYSTEM_SNAPSHOT_FUCHSIA_H_
#define CRASHPAD_SNAPSHOT_FUCHSIA_SYSTEM_SNAPSHOT_FUCHSIA_H_

#include <sys/time.h>

#include "base/macros.h"
#include "build/build_config.h"
#include "snapshot/system_snapshot.h"
#include "util/misc/initialization_state_dcheck.h"

#if defined(ARCH_CPU_X86_FAMILY)
#include "snapshot/x86/cpuid_reader.h"
#endif

namespace crashpad {
namespace internal {

//! \brief A SystemSnapshot of the running system, when the system runs Fuchsia.
class SystemSnapshotFuchsia final : public SystemSnapshot {
 public:
  SystemSnapshotFuchsia();
  ~SystemSnapshotFuchsia() override;

  //! \brief Initializes the object.
  //!
  //! \param[in] snapshot_time The time of the snapshot being taken.
  //!
  //! This parameter is necessary for TimeZone() to determine whether daylight
  //! saving time was in effect at the time the snapshot was taken. Otherwise,
  //! it would need to base its determination on the current time, which may be
  //! different than the snapshot time for snapshots generated around the
  //! daylight saving transition time.
  void Initialize(const timeval* snapshot_time);

  // SystemSnapshot:

  CPUArchitecture GetCPUArchitecture() const override;
  uint32_t CPURevision() const override;
  uint8_t CPUCount() const override;
  std::string CPUVendor() const override;
  void CPUFrequency(uint64_t* current_hz, uint64_t* max_hz) const override;
  uint32_t CPUX86Signature() const override;
  uint64_t CPUX86Features() const override;
  uint64_t CPUX86ExtendedFeatures() const override;
  uint32_t CPUX86Leaf7Features() const override;
  bool CPUX86SupportsDAZ() const override;
  OperatingSystem GetOperatingSystem() const override;
  bool OSServer() const override;
  void OSVersion(
      int* major, int* minor, int* bugfix, std::string* build) const override;
  std::string OSVersionFull() const override;
  bool NXEnabled() const override;
  std::string MachineDescription() const override;
  void TimeZone(DaylightSavingTimeStatus* dst_status,
                int* standard_offset_seconds,
                int* daylight_offset_seconds,
                std::string* standard_name,
                std::string* daylight_name) const override;
 private:
  std::string os_version_full_;
  const timeval* snapshot_time_;  // weak
#if defined(ARCH_CPU_X86_FAMILY)
  CpuidReader cpuid_;
#endif  // ARCH_CPU_X86_FAMILY
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(SystemSnapshotFuchsia);
};

}  // namespace internal
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_FUCHSIA_SYSTEM_SNAPSHOT_FUCHSIA_H_
