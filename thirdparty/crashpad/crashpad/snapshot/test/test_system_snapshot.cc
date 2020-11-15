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

#include "snapshot/test/test_system_snapshot.h"

namespace crashpad {
namespace test {

TestSystemSnapshot::TestSystemSnapshot()
    : cpu_architecture_(kCPUArchitectureUnknown),
      cpu_revision_(0),
      cpu_count_(0),
      cpu_vendor_(),
      cpu_frequency_current_hz_(0),
      cpu_frequency_max_hz_(0),
      cpu_x86_signature_(0),
      cpu_x86_features_(0),
      cpu_x86_extended_features_(0),
      cpu_x86_leaf_7_features_(0),
      cpu_x86_supports_daz_(false),
      operating_system_(kOperatingSystemUnknown),
      os_server_(false),
      os_version_major_(0),
      os_version_minor_(0),
      os_version_bugfix_(0),
      os_version_build_(),
      os_version_full_(),
      nx_enabled_(false),
      machine_description_(),
      time_zone_dst_status_(kDoesNotObserveDaylightSavingTime),
      time_zone_standard_offset_seconds_(0),
      time_zone_daylight_offset_seconds_(0),
      time_zone_standard_name_(),
      time_zone_daylight_name_() {
}

TestSystemSnapshot::~TestSystemSnapshot() {
}

CPUArchitecture TestSystemSnapshot::GetCPUArchitecture() const {
  return cpu_architecture_;
}

uint32_t TestSystemSnapshot::CPURevision() const {
  return cpu_revision_;
}

uint8_t TestSystemSnapshot::CPUCount() const {
  return cpu_count_;
}

std::string TestSystemSnapshot::CPUVendor() const {
  return cpu_vendor_;
}

void TestSystemSnapshot::CPUFrequency(uint64_t* current_hz,
                                      uint64_t* max_hz) const {
  *current_hz = cpu_frequency_current_hz_;
  *max_hz = cpu_frequency_max_hz_;
}

uint32_t TestSystemSnapshot::CPUX86Signature() const {
  return cpu_x86_signature_;
}

uint64_t TestSystemSnapshot::CPUX86Features() const {
  return cpu_x86_features_;
}

uint64_t TestSystemSnapshot::CPUX86ExtendedFeatures() const {
  return cpu_x86_extended_features_;
}

uint32_t TestSystemSnapshot::CPUX86Leaf7Features() const {
  return cpu_x86_leaf_7_features_;
}

bool TestSystemSnapshot::CPUX86SupportsDAZ() const {
  return cpu_x86_supports_daz_;
}

SystemSnapshot::OperatingSystem TestSystemSnapshot::GetOperatingSystem() const {
  return operating_system_;
}

bool TestSystemSnapshot::OSServer() const {
  return os_server_;
}

void TestSystemSnapshot::OSVersion(
    int* major, int* minor, int* bugfix, std::string* build) const {
  *major = os_version_major_;
  *minor = os_version_minor_;
  *bugfix = os_version_bugfix_;
  *build = os_version_build_;
}

std::string TestSystemSnapshot::OSVersionFull() const {
  return os_version_full_;
}

bool TestSystemSnapshot::NXEnabled() const {
  return nx_enabled_;
}

std::string TestSystemSnapshot::MachineDescription() const {
  return machine_description_;
}

void TestSystemSnapshot::TimeZone(DaylightSavingTimeStatus* dst_status,
                                  int* standard_offset_seconds,
                                  int* daylight_offset_seconds,
                                  std::string* standard_name,
                                  std::string* daylight_name) const {
  *dst_status = time_zone_dst_status_;
  *standard_offset_seconds = time_zone_standard_offset_seconds_;
  *daylight_offset_seconds = time_zone_daylight_offset_seconds_;
  *standard_name = time_zone_standard_name_;
  *daylight_name = time_zone_daylight_name_;
}

}  // namespace test
}  // namespace crashpad
