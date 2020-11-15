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

#ifndef CRASHPAD_SNAPSHOT_TEST_TEST_SYSTEM_SNAPSHOT_H_
#define CRASHPAD_SNAPSHOT_TEST_TEST_SYSTEM_SNAPSHOT_H_

#include <stdint.h>

#include <string>

#include "base/macros.h"
#include "snapshot/system_snapshot.h"

namespace crashpad {
namespace test {

//! \brief A test SystemSnapshot that can carry arbitrary data for testing
//!     purposes.
class TestSystemSnapshot final : public SystemSnapshot {
 public:
  TestSystemSnapshot();
  ~TestSystemSnapshot() override;

  void SetCPUArchitecture(CPUArchitecture cpu_architecture) {
    cpu_architecture_ = cpu_architecture;
  }
  void SetCPURevision(uint32_t cpu_revision) { cpu_revision_ = cpu_revision; }
  void SetCPUCount(uint8_t cpu_count) { cpu_count_ = cpu_count; }
  void SetCPUVendor(const std::string& cpu_vendor) { cpu_vendor_ = cpu_vendor; }
  void SetCPUFrequency(uint64_t current_hz, uint64_t max_hz) {
    cpu_frequency_current_hz_ = current_hz;
    cpu_frequency_max_hz_ = max_hz;
  }
  void SetCPUX86Signature(uint32_t cpu_x86_signature) {
    cpu_x86_signature_ = cpu_x86_signature;
  }
  void SetCPUX86Features(uint64_t cpu_x86_features) {
    cpu_x86_features_ = cpu_x86_features;
  }
  void SetCPUX86ExtendedFeatures(uint64_t cpu_x86_extended_features) {
    cpu_x86_extended_features_ = cpu_x86_extended_features;
  }
  void SetCPUX86Leaf7Features(uint32_t cpu_x86_leaf_7_features) {
    cpu_x86_leaf_7_features_ = cpu_x86_leaf_7_features;
  }
  void SetCPUX86SupportsDAZ(bool cpu_x86_supports_daz) {
    cpu_x86_supports_daz_ = cpu_x86_supports_daz;
  }
  void SetOperatingSystem(OperatingSystem operating_system) {
    operating_system_ = operating_system;
  }
  void SetOSServer(bool os_server) { os_server_ = os_server; }
  void SetOSVersion(
      int major, int minor, int bugfix, const std::string& build) {
    os_version_major_ = major;
    os_version_minor_ = minor;
    os_version_bugfix_ = bugfix;
    os_version_build_ = build;
  }
  void SetOSVersionFull(const std::string& os_version_full) {
    os_version_full_ = os_version_full;
  }
  void SetNXEnabled(bool nx_enabled) { nx_enabled_ = nx_enabled; }
  void SetMachineDescription(const std::string& machine_description) {
    machine_description_ = machine_description;
  }
  void SetTimeZone(DaylightSavingTimeStatus dst_status,
                   int standard_offset_seconds,
                   int daylight_offset_seconds,
                   const std::string& standard_name,
                   const std::string& daylight_name) {
    time_zone_dst_status_ = dst_status;
    time_zone_standard_offset_seconds_ = standard_offset_seconds;
    time_zone_daylight_offset_seconds_ = daylight_offset_seconds;
    time_zone_standard_name_ = standard_name;
    time_zone_daylight_name_ = daylight_name;
  }

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
  CPUArchitecture cpu_architecture_;
  uint32_t cpu_revision_;
  uint8_t cpu_count_;
  std::string cpu_vendor_;
  uint64_t cpu_frequency_current_hz_;
  uint64_t cpu_frequency_max_hz_;
  uint32_t cpu_x86_signature_;
  uint64_t cpu_x86_features_;
  uint64_t cpu_x86_extended_features_;
  uint32_t cpu_x86_leaf_7_features_;
  bool cpu_x86_supports_daz_;
  OperatingSystem operating_system_;
  bool os_server_;
  int os_version_major_;
  int os_version_minor_;
  int os_version_bugfix_;
  std::string os_version_build_;
  std::string os_version_full_;
  bool nx_enabled_;
  std::string machine_description_;
  DaylightSavingTimeStatus time_zone_dst_status_;
  int time_zone_standard_offset_seconds_;
  int time_zone_daylight_offset_seconds_;
  std::string time_zone_standard_name_;
  std::string time_zone_daylight_name_;

  DISALLOW_COPY_AND_ASSIGN(TestSystemSnapshot);
};

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_TEST_TEST_SYSTEM_SNAPSHOT_H_
