// Copyright 2014 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_SNAPSHOT_MAC_SYSTEM_SNAPSHOT_MAC_H_
#define CRASHPAD_SNAPSHOT_MAC_SYSTEM_SNAPSHOT_MAC_H_

#include <stdint.h>

#include <string>

#include "base/macros.h"
#include "snapshot/system_snapshot.h"
#include "util/misc/initialization_state_dcheck.h"

namespace crashpad {

class ProcessReaderMac;

namespace internal {

//! \brief A SystemSnapshot of the running system, when the system runs macOS.
class SystemSnapshotMac final : public SystemSnapshot {
 public:
  SystemSnapshotMac();
  ~SystemSnapshotMac() override;

  //! \brief Initializes the object.
  //!
  //! \param[in] process_reader A reader for the process being snapshotted.
  //!     \n\n
  //!     It seems odd that a system snapshot implementation would need a
  //!     ProcessReaderMac, but some of the information reported about the
  //!     system depends on the process it’s being reported for. For example,
  //!     the architecture returned by GetCPUArchitecture() should be the
  //!     architecture of the process, which may be different than the native
  //!     architecture of the system: an x86_64 system can run both x86_64 and
  //!     32-bit x86 processes.
  //! \param[in] snapshot_time The time of the snapshot being taken.
  //!     \n\n
  //!     This parameter is necessary for TimeZone() to determine whether
  //!     daylight saving time was in effect at the time the snapshot was taken.
  //!     Otherwise, it would need to base its determination on the current
  //!     time, which may be different than the snapshot time for snapshots
  //!     generated around the daylight saving transition time.
  void Initialize(ProcessReaderMac* process_reader,
                  const timeval* snapshot_time);

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
  std::string os_version_build_;
  ProcessReaderMac* process_reader_;  // weak
  const timeval* snapshot_time_;  // weak
  int os_version_major_;
  int os_version_minor_;
  int os_version_bugfix_;
  bool os_server_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(SystemSnapshotMac);
};

}  // namespace internal
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_MAC_SYSTEM_SNAPSHOT_MAC_H_
