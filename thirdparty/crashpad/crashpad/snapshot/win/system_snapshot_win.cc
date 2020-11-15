// Copyright 2015 The Crashpad Authors. All rights reserved.
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

#include "snapshot/win/system_snapshot_win.h"

#include <windows.h>
#include <intrin.h>
#include <powrprof.h>
#include <winnt.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "base/numerics/safe_conversions.h"
#include "base/strings/stringprintf.h"
#include "base/strings/utf_string_conversions.h"
#include "util/win/module_version.h"

namespace crashpad {

namespace {

//! \brief Gets a string representation for a VS_FIXEDFILEINFO.dwFileFlags
//!     value.
std::string GetStringForFileFlags(uint32_t file_flags) {
  std::string result;
  DCHECK_EQ(file_flags & VS_FF_INFOINFERRED, 0u);
  if (file_flags & VS_FF_DEBUG)
    result += "Debug,";
  if (file_flags & VS_FF_PATCHED)
    result += "Patched,";
  if (file_flags & VS_FF_PRERELEASE)
    result += "Prerelease,";
  if (file_flags & VS_FF_PRIVATEBUILD)
    result += "Private,";
  if (file_flags & VS_FF_SPECIALBUILD)
    result += "Special,";
  if (!result.empty())
    return result.substr(0, result.size() - 1);  // Remove trailing comma.
  return result;
}

//! \brief Gets a string representation for a VS_FIXEDFILEINFO.dwFileOS value.
std::string GetStringForFileOS(uint32_t file_os) {
  // There are a variety of ancient things this could theoretically be. In
  // practice, we're always going to get VOS_NT_WINDOWS32 here.
  if ((file_os & VOS_NT_WINDOWS32) == VOS_NT_WINDOWS32)
    return "Windows NT";
  else
    return "Unknown";
}

}  // namespace

namespace internal {

SystemSnapshotWin::SystemSnapshotWin()
    : SystemSnapshot(),
      os_version_full_(),
      os_version_build_(),
      process_reader_(nullptr),
      os_version_major_(0),
      os_version_minor_(0),
      os_version_bugfix_(0),
      os_server_(false),
      initialized_() {
}

SystemSnapshotWin::~SystemSnapshotWin() {
}

void SystemSnapshotWin::Initialize(ProcessReaderWin* process_reader) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  process_reader_ = process_reader;

  // We use both GetVersionEx() and GetModuleVersionAndType() (which uses
  // VerQueryValue() internally). GetVersionEx() is not trustworthy after
  // Windows 8 (depending on the application manifest) so its data is used only
  // to fill the os_server_ field, and the rest comes from the version
  // information stamped on kernel32.dll.
  OSVERSIONINFOEX version_info = {sizeof(version_info)};
  if (!GetVersionEx(reinterpret_cast<OSVERSIONINFO*>(&version_info))) {
    PLOG(WARNING) << "GetVersionEx";
  } else {
    os_server_ = version_info.wProductType != VER_NT_WORKSTATION;
  }

  static constexpr wchar_t kSystemDll[] = L"kernel32.dll";
  VS_FIXEDFILEINFO ffi;
  if (GetModuleVersionAndType(base::FilePath(kSystemDll), &ffi)) {
    std::string flags_string = GetStringForFileFlags(ffi.dwFileFlags);
    std::string os_name = GetStringForFileOS(ffi.dwFileOS);
    os_version_major_ = ffi.dwFileVersionMS >> 16;
    os_version_minor_ = ffi.dwFileVersionMS & 0xffff;
    os_version_bugfix_ = ffi.dwFileVersionLS >> 16;
    os_version_build_ = base::StringPrintf("%lu", ffi.dwFileVersionLS & 0xffff);
    os_version_full_ = base::StringPrintf(
        "%s %u.%u.%u.%s%s",
        os_name.c_str(),
        os_version_major_,
        os_version_minor_,
        os_version_bugfix_,
        os_version_build_.c_str(),
        flags_string.empty()
            ? ""
            : (std::string(" (") + flags_string + ")").c_str());
  }

  INITIALIZATION_STATE_SET_VALID(initialized_);
}

CPUArchitecture SystemSnapshotWin::GetCPUArchitecture() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  return process_reader_->Is64Bit() ? kCPUArchitectureX86_64
                                    : kCPUArchitectureX86;
}

uint32_t SystemSnapshotWin::CPURevision() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  uint32_t raw = CPUX86Signature();
  uint8_t stepping = raw & 0xf;
  uint8_t model = (raw & 0xf0) >> 4;
  uint8_t family = (raw & 0xf00) >> 8;
  uint8_t extended_model = static_cast<uint8_t>((raw & 0xf0000) >> 16);
  uint16_t extended_family = (raw & 0xff00000) >> 20;

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

uint8_t SystemSnapshotWin::CPUCount() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  SYSTEM_INFO system_info;
  GetSystemInfo(&system_info);
  if (!base::IsValueInRangeForNumericType<uint8_t>(
          system_info.dwNumberOfProcessors)) {
    LOG(WARNING) << "dwNumberOfProcessors exceeds uint8_t storage";
  }
  return base::saturated_cast<uint8_t>(system_info.dwNumberOfProcessors);
}

std::string SystemSnapshotWin::CPUVendor() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  int cpu_info[4];
  __cpuid(cpu_info, 0);
  char vendor[12];
  *reinterpret_cast<int*>(vendor) = cpu_info[1];
  *reinterpret_cast<int*>(vendor + 4) = cpu_info[3];
  *reinterpret_cast<int*>(vendor + 8) = cpu_info[2];
  return std::string(vendor, sizeof(vendor));
}

void SystemSnapshotWin::CPUFrequency(uint64_t* current_hz,
                                     uint64_t* max_hz) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  int num_cpus = CPUCount();
  DCHECK_GT(num_cpus, 0);
  std::vector<PROCESSOR_POWER_INFORMATION> info(num_cpus);
  if (CallNtPowerInformation(ProcessorInformation,
                             nullptr,
                             0,
                             &info[0],
                             sizeof(PROCESSOR_POWER_INFORMATION) * num_cpus) !=
      0) {
    *current_hz = 0;
    *max_hz = 0;
    return;
  }
  constexpr uint64_t kMhzToHz = static_cast<uint64_t>(1E6);
  *current_hz = std::max_element(info.begin(),
                                 info.end(),
                                 [](const PROCESSOR_POWER_INFORMATION& a,
                                    const PROCESSOR_POWER_INFORMATION& b) {
                                   return a.CurrentMhz < b.CurrentMhz;
                                 })->CurrentMhz *
                kMhzToHz;
  *max_hz = std::max_element(info.begin(),
                             info.end(),
                             [](const PROCESSOR_POWER_INFORMATION& a,
                                const PROCESSOR_POWER_INFORMATION& b) {
                               return a.MaxMhz < b.MaxMhz;
                             })->MaxMhz *
            kMhzToHz;
}

uint32_t SystemSnapshotWin::CPUX86Signature() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  int cpu_info[4];
  // We will never run on any processors that don't support at least function 1.
  __cpuid(cpu_info, 1);
  return cpu_info[0];
}

uint64_t SystemSnapshotWin::CPUX86Features() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  int cpu_info[4];
  // We will never run on any processors that don't support at least function 1.
  __cpuid(cpu_info, 1);
  return (static_cast<uint64_t>(cpu_info[2]) << 32) |
         static_cast<uint64_t>(cpu_info[3]);
}

uint64_t SystemSnapshotWin::CPUX86ExtendedFeatures() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  int cpu_info[4];
  // We will never run on any processors that don't support at least extended
  // function 1.
  __cpuid(cpu_info, 0x80000001);
  return (static_cast<uint64_t>(cpu_info[2]) << 32) |
         static_cast<uint64_t>(cpu_info[3]);
}

uint32_t SystemSnapshotWin::CPUX86Leaf7Features() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  int cpu_info[4];

  // Make sure leaf 7 can be called.
  __cpuid(cpu_info, 0);
  if (cpu_info[0] < 7)
    return 0;

  __cpuidex(cpu_info, 7, 0);
  return cpu_info[1];
}

bool SystemSnapshotWin::CPUX86SupportsDAZ() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  // The correct way to check for denormals-as-zeros (DAZ) support is to examine
  // mxcsr mask, which can be done with fxsave. See Intel Software Developer's
  // Manual, Volume 1: Basic Architecture (253665-051), 11.6.3 "Checking for the
  // DAZ Flag in the MXCSR Register". Note that since this function tests for
  // DAZ support in the CPU, it checks the mxcsr mask. Testing mxcsr would
  // indicate whether DAZ is actually enabled, which is a per-thread context
  // concern.

  // Test for fxsave support.
  uint64_t features = CPUX86Features();
  if (!(features & (UINT64_C(1) << 24))) {
    return false;
  }

  // Call fxsave.
  __declspec(align(16)) uint32_t extended_registers[128];
  _fxsave(&extended_registers);
  uint32_t mxcsr_mask = extended_registers[7];

  // Test the DAZ bit.
  return (mxcsr_mask & (1 << 6)) != 0;
}

SystemSnapshot::OperatingSystem SystemSnapshotWin::GetOperatingSystem() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return kOperatingSystemWindows;
}

bool SystemSnapshotWin::OSServer() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return os_server_;
}

void SystemSnapshotWin::OSVersion(int* major,
                                  int* minor,
                                  int* bugfix,
                                  std::string* build) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  *major = os_version_major_;
  *minor = os_version_minor_;
  *bugfix = os_version_bugfix_;
  build->assign(os_version_build_);
}

std::string SystemSnapshotWin::OSVersionFull() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return os_version_full_;
}

std::string SystemSnapshotWin::MachineDescription() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  // TODO(scottmg): Not sure if there's anything sensible to put here.
  return std::string();
}

bool SystemSnapshotWin::NXEnabled() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return !!IsProcessorFeaturePresent(PF_NX_ENABLED);
}

void SystemSnapshotWin::TimeZone(DaylightSavingTimeStatus* dst_status,
                                 int* standard_offset_seconds,
                                 int* daylight_offset_seconds,
                                 std::string* standard_name,
                                 std::string* daylight_name) const {
  // This returns the current time zone status rather than the status at the
  // time of the snapshot. This differs from the Mac implementation.
  TIME_ZONE_INFORMATION time_zone_information;
  *dst_status = static_cast<DaylightSavingTimeStatus>(
      GetTimeZoneInformation(&time_zone_information));
  *standard_offset_seconds =
      (time_zone_information.Bias + time_zone_information.StandardBias) * -60;
  *daylight_offset_seconds =
      (time_zone_information.Bias + time_zone_information.DaylightBias) * -60;
  *standard_name = base::UTF16ToUTF8(time_zone_information.StandardName);
  *daylight_name = base::UTF16ToUTF8(time_zone_information.DaylightName);
}

}  // namespace internal
}  // namespace crashpad
