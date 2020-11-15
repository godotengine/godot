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

#include "snapshot/linux/system_snapshot_linux.h"

#include <stddef.h>
#include <sys/types.h>
#include <sys/utsname.h>

#include <algorithm>

#include "base/files/file_path.h"
#include "base/logging.h"
#include "base/strings/string_number_conversions.h"
#include "base/strings/string_piece.h"
#include "base/strings/stringprintf.h"
#include "snapshot/cpu_context.h"
#include "snapshot/posix/timezone.h"
#include "util/file/file_io.h"
#include "util/numeric/in_range_cast.h"
#include "util/string/split_string.h"

#if defined(OS_ANDROID)
#include <sys/system_properties.h>
#endif

namespace crashpad {
namespace internal {

namespace {

bool ReadCPUsOnline(uint32_t* first_cpu, uint8_t* cpu_count) {
  std::string contents;
  if (!LoggingReadEntireFile(base::FilePath("/sys/devices/system/cpu/online"),
                             &contents)) {
    return false;
  }
  if (contents.back() != '\n') {
    LOG(ERROR) << "format error";
    return false;
  }
  contents.pop_back();

  unsigned int count = 0;
  unsigned int first = 0;
  bool have_first = false;
  std::vector<std::string> ranges = SplitString(contents, ',');
  for (const auto& range : ranges) {
    std::string left, right;
    if (SplitStringFirst(range, '-', &left, &right)) {
      unsigned int start, end;
      if (!StringToUint(base::StringPiece(left), &start) ||
          !StringToUint(base::StringPiece(right), &end) || end <= start) {
        LOG(ERROR) << "format error: " << range;
        return false;
      }
      if (end <= start) {
        LOG(ERROR) << "format error";
        return false;
      }
      count += end - start + 1;
      if (!have_first) {
        first = start;
        have_first = true;
      }
    } else {
      unsigned int cpuno;
      if (!StringToUint(base::StringPiece(range), &cpuno)) {
        LOG(ERROR) << "format error";
        return false;
      }
      if (!have_first) {
        first = cpuno;
        have_first = true;
      }
      ++count;
    }
  }
  if (!have_first) {
    LOG(ERROR) << "no cpus online";
    return false;
  }
  *cpu_count = InRangeCast<uint8_t>(count, std::numeric_limits<uint8_t>::max());
  *first_cpu = first;
  return true;
}

bool ReadFreqFile(const std::string& filename, uint64_t* hz) {
  std::string contents;
  if (!LoggingReadEntireFile(base::FilePath(filename), &contents)) {
    return false;
  }
  if (contents.back() != '\n') {
    LOG(ERROR) << "format error";
    return false;
  }
  contents.pop_back();

  uint64_t khz;
  if (!base::StringToUint64(base::StringPiece(contents), &khz)) {
    LOG(ERROR) << "format error";
    return false;
  }

  *hz = khz * 1000;
  return true;
}

#if defined(OS_ANDROID)
bool ReadProperty(const char* property, std::string* value) {
  char value_buffer[PROP_VALUE_MAX];
  int length = __system_property_get(property, value_buffer);
  if (length <= 0) {
    LOG(ERROR) << "Couldn't read property " << property;
    return false;
  }
  *value = value_buffer;
  return true;
}
#endif  // OS_ANDROID

}  // namespace

SystemSnapshotLinux::SystemSnapshotLinux()
    : SystemSnapshot(),
      os_version_full_(),
      os_version_build_(),
      process_reader_(nullptr),
      snapshot_time_(nullptr),
#if defined(ARCH_CPU_X86_FAMILY)
      cpuid_(),
#endif  // ARCH_CPU_X86_FAMILY
      os_version_major_(-1),
      os_version_minor_(-1),
      os_version_bugfix_(-1),
      target_cpu_(0),
      cpu_count_(0),
      initialized_() {
}

SystemSnapshotLinux::~SystemSnapshotLinux() {}

void SystemSnapshotLinux::Initialize(ProcessReaderLinux* process_reader,
                                     const timeval* snapshot_time) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);
  process_reader_ = process_reader;
  snapshot_time_ = snapshot_time;

#if defined(OS_ANDROID)
  std::string build_string;
  if (ReadProperty("ro.build.fingerprint", &build_string)) {
    os_version_build_ = build_string;
    os_version_full_ = build_string;
  }
#endif  // OS_ANDROID

  utsname uts;
  if (uname(&uts) != 0) {
    PLOG(WARNING) << "uname";
  } else {
    if (!os_version_full_.empty()) {
      os_version_full_.push_back(' ');
    }
    os_version_full_ += base::StringPrintf(
        "%s %s %s %s", uts.sysname, uts.release, uts.version, uts.machine);
  }
  ReadKernelVersion(uts.release);

  if (!os_version_build_.empty()) {
    os_version_build_.push_back(' ');
  }
  os_version_build_ += uts.version;
  os_version_build_.push_back(' ');
  os_version_build_ += uts.machine;

  if (!ReadCPUsOnline(&target_cpu_, &cpu_count_)) {
    target_cpu_ = 0;
    cpu_count_ = 0;
  }

  INITIALIZATION_STATE_SET_VALID(initialized_);
}

CPUArchitecture SystemSnapshotLinux::GetCPUArchitecture() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
#if defined(ARCH_CPU_X86_FAMILY)
  return process_reader_->Is64Bit() ? kCPUArchitectureX86_64
                                    : kCPUArchitectureX86;
#elif defined(ARCH_CPU_ARM_FAMILY)
  return process_reader_->Is64Bit() ? kCPUArchitectureARM64
                                    : kCPUArchitectureARM;
#elif defined(ARCH_CPU_MIPS_FAMILY)
  return process_reader_->Is64Bit() ? kCPUArchitectureMIPS64EL
                                    : kCPUArchitectureMIPSEL;
#else
#error port to your architecture
#endif
}

uint32_t SystemSnapshotLinux::CPURevision() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
#if defined(ARCH_CPU_X86_FAMILY)
  return cpuid_.Revision();
#elif defined(ARCH_CPU_ARM_FAMILY)
  // TODO(jperaza): do this. https://crashpad.chromium.org/bug/30
  return 0;
#elif defined(ARCH_CPU_MIPS_FAMILY)
  // Not implementable on MIPS
  return 0;
#else
#error port to your architecture
#endif
}

uint8_t SystemSnapshotLinux::CPUCount() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return cpu_count_;
}

std::string SystemSnapshotLinux::CPUVendor() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
#if defined(ARCH_CPU_X86_FAMILY)
  return cpuid_.Vendor();
#elif defined(ARCH_CPU_ARM_FAMILY)
  // TODO(jperaza): do this. https://crashpad.chromium.org/bug/30
  return std::string();
#elif defined(ARCH_CPU_MIPS_FAMILY)
  // Not implementable on MIPS
  return std::string();
#else
#error port to your architecture
#endif
}

void SystemSnapshotLinux::CPUFrequency(uint64_t* current_hz,
                                       uint64_t* max_hz) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  *current_hz = 0;
  *max_hz = 0;

  ReadFreqFile(base::StringPrintf(
                   "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq",
                   target_cpu_),
               current_hz);

  ReadFreqFile(base::StringPrintf(
                   "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_max_freq",
                   target_cpu_),
               max_hz);
}

uint32_t SystemSnapshotLinux::CPUX86Signature() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
#if defined(ARCH_CPU_X86_FAMILY)
  return cpuid_.Signature();
#else
  NOTREACHED();
  return 0;
#endif
}

uint64_t SystemSnapshotLinux::CPUX86Features() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
#if defined(ARCH_CPU_X86_FAMILY)
  return cpuid_.Features();
#else
  NOTREACHED();
  return 0;
#endif
}

uint64_t SystemSnapshotLinux::CPUX86ExtendedFeatures() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
#if defined(ARCH_CPU_X86_FAMILY)
  return cpuid_.ExtendedFeatures();
#else
  NOTREACHED();
  return 0;
#endif
}

uint32_t SystemSnapshotLinux::CPUX86Leaf7Features() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
#if defined(ARCH_CPU_X86_FAMILY)
  return cpuid_.Leaf7Features();
#else
  NOTREACHED();
  return 0;
#endif
}

bool SystemSnapshotLinux::CPUX86SupportsDAZ() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
#if defined(ARCH_CPU_X86_FAMILY)
  return cpuid_.SupportsDAZ();
#else
  NOTREACHED();
  return false;
#endif  // ARCH_CPU_X86_FMAILY
}

SystemSnapshot::OperatingSystem SystemSnapshotLinux::GetOperatingSystem()
    const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
#if defined(OS_ANDROID)
  return kOperatingSystemAndroid;
#else
  return kOperatingSystemLinux;
#endif  // OS_ANDROID
}

bool SystemSnapshotLinux::OSServer() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return false;
}

void SystemSnapshotLinux::OSVersion(int* major,
                                    int* minor,
                                    int* bugfix,
                                    std::string* build) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  *major = os_version_major_;
  *minor = os_version_minor_;
  *bugfix = os_version_bugfix_;
  build->assign(os_version_build_);
}

std::string SystemSnapshotLinux::OSVersionFull() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return os_version_full_;
}

std::string SystemSnapshotLinux::MachineDescription() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
#if defined(OS_ANDROID)
  std::string description;
  std::string prop;
  if (ReadProperty("ro.product.model", &prop)) {
    description += prop;
  }
  if (ReadProperty("ro.product.board", &prop)) {
    if (!description.empty()) {
      description.push_back(' ');
    }
    description += prop;
  }
  return description;
#else
  return std::string();
#endif  // OS_ANDROID
}

bool SystemSnapshotLinux::NXEnabled() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
#if defined(ARCH_CPU_X86_FAMILY)
  return cpuid_.NXEnabled();
#elif defined(ARCH_CPU_ARM_FAMILY)
  // TODO(jperaza): do this. https://crashpad.chromium.org/bug/30
  return false;
#elif defined(ARCH_CPU_MIPS_FAMILY)
  // Not implementable on MIPS
  return false;
#else
#error Port.
#endif  // ARCH_CPU_X86_FAMILY
}

void SystemSnapshotLinux::TimeZone(DaylightSavingTimeStatus* dst_status,
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

void SystemSnapshotLinux::ReadKernelVersion(const std::string& version_string) {
  std::vector<std::string> versions = SplitString(version_string, '.');
  if (versions.size() < 3) {
    LOG(WARNING) << "format error";
    return;
  }

  if (!StringToInt(base::StringPiece(versions[0]), &os_version_major_)) {
    LOG(WARNING) << "no kernel version";
    return;
  }
  DCHECK_GE(os_version_major_, 3);

  if (!StringToInt(base::StringPiece(versions[1]), &os_version_minor_)) {
    LOG(WARNING) << "no major revision";
    return;
  }
  DCHECK_GE(os_version_minor_, 0);

  size_t minor_rev_end = versions[2].find_first_not_of("0123456789");
  if (minor_rev_end == std::string::npos) {
    minor_rev_end = versions[2].size();
  }
  if (!StringToInt(base::StringPiece(versions[2].c_str(), minor_rev_end),
                   &os_version_bugfix_)) {
    LOG(WARNING) << "no minor revision";
    return;
  }
  DCHECK_GE(os_version_bugfix_, 0);

  if (!os_version_build_.empty()) {
    os_version_build_.push_back(' ');
  }
  os_version_build_ += versions[2].substr(minor_rev_end);
}

}  // namespace internal
}  // namespace crashpad
