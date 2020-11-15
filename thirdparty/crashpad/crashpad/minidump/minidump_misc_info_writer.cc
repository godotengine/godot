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

#include "minidump/minidump_misc_info_writer.h"

#include <limits>

#include "base/logging.h"
#include "base/numerics/safe_conversions.h"
#include "base/strings/stringprintf.h"
#include "base/strings/utf_string_conversions.h"
#include "build/build_config.h"
#include "minidump/minidump_writer_util.h"
#include "package.h"
#include "snapshot/process_snapshot.h"
#include "snapshot/system_snapshot.h"
#include "util/file/file_writer.h"
#include "util/misc/arraysize_unsafe.h"
#include "util/numeric/in_range_cast.h"
#include "util/numeric/safe_assignment.h"

#if defined(OS_MACOSX)
#include <AvailabilityMacros.h>
#elif defined(OS_ANDROID)
#include <android/api-level.h>
#endif

namespace crashpad {
namespace {

uint32_t TimevalToRoundedSeconds(const timeval& tv) {
  uint32_t seconds =
      InRangeCast<uint32_t>(tv.tv_sec, std::numeric_limits<uint32_t>::max());
  constexpr int kMicrosecondsPerSecond = static_cast<int>(1E6);
  if (tv.tv_usec >= kMicrosecondsPerSecond / 2 &&
      seconds != std::numeric_limits<uint32_t>::max()) {
    ++seconds;
  }
  return seconds;
}

// For MINIDUMP_MISC_INFO_4::BuildString. dbghelp only places OS version
// information here, but if a machine description is also available, this is the
// only reasonable place in a minidump file to put it.
std::string BuildString(const SystemSnapshot* system_snapshot) {
  std::string os_version_full = system_snapshot->OSVersionFull();
  std::string machine_description = system_snapshot->MachineDescription();
  if (!os_version_full.empty()) {
    if (!machine_description.empty()) {
      return base::StringPrintf(
          "%s; %s", os_version_full.c_str(), machine_description.c_str());
    }
    return os_version_full;
  }
  return machine_description;
}

#if defined(OS_MACOSX)
// Converts the value of the MAC_OS_VERSION_MIN_REQUIRED or
// MAC_OS_X_VERSION_MAX_ALLOWED macro from <AvailabilityMacros.h> to a number
// identifying the minor macOS version that it represents. For example, with an
// argument of MAC_OS_X_VERSION_10_6, this function will return 6.
int AvailabilityVersionToMacOSXMinorVersion(int availability) {
  // Through MAC_OS_X_VERSION_10_9, the minor version is the tens digit.
  if (availability >= 1000 && availability <= 1099) {
    return (availability / 10) % 10;
  }

  // After MAC_OS_X_VERSION_10_9, the older format was insufficient to represent
  // versions. Since then, the minor version is the thousands and hundreds
  // digits.
  if (availability >= 100000 && availability <= 109999) {
    return (availability / 100) % 100;
  }

  return 0;
}
#endif

}  // namespace

namespace internal {

// For MINIDUMP_MISC_INFO_4::DbgBldStr. dbghelp produces strings like
// “dbghelp.i386,6.3.9600.16520” and “dbghelp.amd64,6.3.9600.16520”. Mimic that
// format, and add the OS that wrote the minidump along with any relevant
// platform-specific data describing the compilation environment.
std::string MinidumpMiscInfoDebugBuildString() {
  // Caution: the minidump file format only has room for 39 UTF-16 code units
  // plus a UTF-16 NUL terminator. Don’t let strings get longer than this, or
  // they will be truncated and a message will be logged.
#if defined(OS_MACOSX)
  static constexpr char kOS[] = "mac";
#elif defined(OS_ANDROID)
  static constexpr char kOS[] = "android";
#elif defined(OS_LINUX)
  static constexpr char kOS[] = "linux";
#elif defined(OS_WIN)
  static constexpr char kOS[] = "win";
#elif defined(OS_FUCHSIA)
  static constexpr char kOS[] = "fuchsia";
#else
#error define kOS for this operating system
#endif

#if defined(ARCH_CPU_X86)
  static constexpr char kCPU[] = "i386";
#elif defined(ARCH_CPU_X86_64)
  static constexpr char kCPU[] = "amd64";
#elif defined(ARCH_CPU_ARMEL)
  static constexpr char kCPU[] = "arm";
#elif defined(ARCH_CPU_ARM64)
  static constexpr char kCPU[] = "arm64";
#elif defined(ARCH_CPU_MIPSEL)
  static constexpr char kCPU[] = "mips";
#elif defined(ARCH_CPU_MIPS64EL)
  static constexpr char kCPU[] = "mips64";
#else
#error define kCPU for this CPU
#endif

  std::string debug_build_string = base::StringPrintf("%s.%s,%s,%s",
                                                      PACKAGE_TARNAME,
                                                      kCPU,
                                                      PACKAGE_VERSION,
                                                      kOS);

#if defined(OS_MACOSX)
  debug_build_string += base::StringPrintf(
      ",%d,%d",
      AvailabilityVersionToMacOSXMinorVersion(MAC_OS_X_VERSION_MIN_REQUIRED),
      AvailabilityVersionToMacOSXMinorVersion(MAC_OS_X_VERSION_MAX_ALLOWED));
#elif defined(OS_ANDROID)
  debug_build_string += base::StringPrintf(",%d", __ANDROID_API__);
#endif

  return debug_build_string;
}

}  // namespace internal

MinidumpMiscInfoWriter::MinidumpMiscInfoWriter()
    : MinidumpStreamWriter(), misc_info_(), has_xstate_data_(false) {
}

MinidumpMiscInfoWriter::~MinidumpMiscInfoWriter() {
}

void MinidumpMiscInfoWriter::InitializeFromSnapshot(
    const ProcessSnapshot* process_snapshot) {
  DCHECK_EQ(state(), kStateMutable);
  DCHECK_EQ(misc_info_.Flags1, 0u);

  SetProcessID(InRangeCast<uint32_t>(process_snapshot->ProcessID(), 0));

  const SystemSnapshot* system_snapshot = process_snapshot->System();

  uint64_t current_hz;
  uint64_t max_hz;
  system_snapshot->CPUFrequency(&current_hz, &max_hz);
  constexpr uint32_t kHzPerMHz = static_cast<const uint32_t>(1E6);
  SetProcessorPowerInfo(
      InRangeCast<uint32_t>(current_hz / kHzPerMHz,
                            std::numeric_limits<uint32_t>::max()),
      InRangeCast<uint32_t>(max_hz / kHzPerMHz,
                            std::numeric_limits<uint32_t>::max()),
      0,
      0,
      0);

  timeval start_time;
  process_snapshot->ProcessStartTime(&start_time);

  timeval user_time;
  timeval system_time;
  process_snapshot->ProcessCPUTimes(&user_time, &system_time);

  // Round the resource usage fields to the nearest second, because the minidump
  // format only has one-second resolution. The start_time field is truncated
  // instead of rounded so that the process uptime is reflected more accurately
  // when the start time is compared to the snapshot time in the
  // MINIDUMP_HEADER, which is also truncated, not rounded.
  uint32_t user_seconds = TimevalToRoundedSeconds(user_time);
  uint32_t system_seconds = TimevalToRoundedSeconds(system_time);

  SetProcessTimes(start_time.tv_sec, user_seconds, system_seconds);

  // This determines the system’s time zone, which may be different than the
  // process’ notion of the time zone.
  SystemSnapshot::DaylightSavingTimeStatus dst_status;
  int standard_offset_seconds;
  int daylight_offset_seconds;
  std::string standard_name;
  std::string daylight_name;
  system_snapshot->TimeZone(&dst_status,
                            &standard_offset_seconds,
                            &daylight_offset_seconds,
                            &standard_name,
                            &daylight_name);

  // standard_offset_seconds is seconds east of UTC, but the minidump file wants
  // minutes west of UTC. daylight_offset_seconds is also seconds east of UTC,
  // but the minidump file wants minutes west of the standard offset. The empty
  // ({}) arguments are for the transition times in and out of daylight saving
  // time. These are not determined because no API exists to do so, and the
  // transition times may vary from year to year.
  SetTimeZone(dst_status,
              standard_offset_seconds / -60,
              standard_name,
              {},
              0,
              daylight_name,
              {},
              (standard_offset_seconds - daylight_offset_seconds) / 60);

  SetBuildString(BuildString(system_snapshot),
                 internal::MinidumpMiscInfoDebugBuildString());
}

void MinidumpMiscInfoWriter::SetProcessID(uint32_t process_id) {
  DCHECK_EQ(state(), kStateMutable);

  misc_info_.ProcessId = process_id;
  misc_info_.Flags1 |= MINIDUMP_MISC1_PROCESS_ID;
}

void MinidumpMiscInfoWriter::SetProcessTimes(time_t process_create_time,
                                             uint32_t process_user_time,
                                             uint32_t process_kernel_time) {
  DCHECK_EQ(state(), kStateMutable);

  internal::MinidumpWriterUtil::AssignTimeT(&misc_info_.ProcessCreateTime,
                                            process_create_time);

  misc_info_.ProcessUserTime = process_user_time;
  misc_info_.ProcessKernelTime = process_kernel_time;
  misc_info_.Flags1 |= MINIDUMP_MISC1_PROCESS_TIMES;
}

void MinidumpMiscInfoWriter::SetProcessorPowerInfo(
    uint32_t processor_max_mhz,
    uint32_t processor_current_mhz,
    uint32_t processor_mhz_limit,
    uint32_t processor_max_idle_state,
    uint32_t processor_current_idle_state) {
  DCHECK_EQ(state(), kStateMutable);

  misc_info_.ProcessorMaxMhz = processor_max_mhz;
  misc_info_.ProcessorCurrentMhz = processor_current_mhz;
  misc_info_.ProcessorMhzLimit = processor_mhz_limit;
  misc_info_.ProcessorMaxIdleState = processor_max_idle_state;
  misc_info_.ProcessorCurrentIdleState = processor_current_idle_state;
  misc_info_.Flags1 |= MINIDUMP_MISC1_PROCESSOR_POWER_INFO;
}

void MinidumpMiscInfoWriter::SetProcessIntegrityLevel(
    uint32_t process_integrity_level) {
  DCHECK_EQ(state(), kStateMutable);

  misc_info_.ProcessIntegrityLevel = process_integrity_level;
  misc_info_.Flags1 |= MINIDUMP_MISC3_PROCESS_INTEGRITY;
}

void MinidumpMiscInfoWriter::SetProcessExecuteFlags(
    uint32_t process_execute_flags) {
  DCHECK_EQ(state(), kStateMutable);

  misc_info_.ProcessExecuteFlags = process_execute_flags;
  misc_info_.Flags1 |= MINIDUMP_MISC3_PROCESS_EXECUTE_FLAGS;
}

void MinidumpMiscInfoWriter::SetProtectedProcess(uint32_t protected_process) {
  DCHECK_EQ(state(), kStateMutable);

  misc_info_.ProtectedProcess = protected_process;
  misc_info_.Flags1 |= MINIDUMP_MISC3_PROTECTED_PROCESS;
}

void MinidumpMiscInfoWriter::SetTimeZone(uint32_t time_zone_id,
                                         int32_t bias,
                                         const std::string& standard_name,
                                         const SYSTEMTIME& standard_date,
                                         int32_t standard_bias,
                                         const std::string& daylight_name,
                                         const SYSTEMTIME& daylight_date,
                                         int32_t daylight_bias) {
  DCHECK_EQ(state(), kStateMutable);

  misc_info_.TimeZoneId = time_zone_id;
  misc_info_.TimeZone.Bias = bias;

  internal::MinidumpWriterUtil::AssignUTF8ToUTF16(
      misc_info_.TimeZone.StandardName,
      ARRAYSIZE_UNSAFE(misc_info_.TimeZone.StandardName),
      standard_name);

  misc_info_.TimeZone.StandardDate = standard_date;
  misc_info_.TimeZone.StandardBias = standard_bias;

  internal::MinidumpWriterUtil::AssignUTF8ToUTF16(
      misc_info_.TimeZone.DaylightName,
      ARRAYSIZE_UNSAFE(misc_info_.TimeZone.DaylightName),
      daylight_name);

  misc_info_.TimeZone.DaylightDate = daylight_date;
  misc_info_.TimeZone.DaylightBias = daylight_bias;

  misc_info_.Flags1 |= MINIDUMP_MISC3_TIMEZONE;
}

void MinidumpMiscInfoWriter::SetBuildString(
    const std::string& build_string,
    const std::string& debug_build_string) {
  DCHECK_EQ(state(), kStateMutable);

  misc_info_.Flags1 |= MINIDUMP_MISC4_BUILDSTRING;

  internal::MinidumpWriterUtil::AssignUTF8ToUTF16(
      misc_info_.BuildString,
      ARRAYSIZE_UNSAFE(misc_info_.BuildString),
      build_string);
  internal::MinidumpWriterUtil::AssignUTF8ToUTF16(
      misc_info_.DbgBldStr,
      ARRAYSIZE_UNSAFE(misc_info_.DbgBldStr),
      debug_build_string);
}

void MinidumpMiscInfoWriter::SetXStateData(
    const XSTATE_CONFIG_FEATURE_MSC_INFO& xstate_data) {
  DCHECK_EQ(state(), kStateMutable);

  misc_info_.XStateData = xstate_data;
  has_xstate_data_ = true;
}

void MinidumpMiscInfoWriter::SetProcessCookie(uint32_t process_cookie) {
  DCHECK_EQ(state(), kStateMutable);

  misc_info_.ProcessCookie = process_cookie;
  misc_info_.Flags1 |= MINIDUMP_MISC5_PROCESS_COOKIE;
}

bool MinidumpMiscInfoWriter::Freeze() {
  DCHECK_EQ(state(), kStateMutable);

  if (!MinidumpStreamWriter::Freeze()) {
    return false;
  }

  size_t size = CalculateSizeOfObjectFromFlags();
  if (!AssignIfInRange(&misc_info_.SizeOfInfo, size)) {
    LOG(ERROR) << "size " << size << " out of range";
    return false;
  }

  return true;
}

size_t MinidumpMiscInfoWriter::SizeOfObject() {
  DCHECK_GE(state(), kStateFrozen);

  return CalculateSizeOfObjectFromFlags();
}

bool MinidumpMiscInfoWriter::WriteObject(FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);

  return file_writer->Write(&misc_info_, CalculateSizeOfObjectFromFlags());
}

MinidumpStreamType MinidumpMiscInfoWriter::StreamType() const {
  return kMinidumpStreamTypeMiscInfo;
}

size_t MinidumpMiscInfoWriter::CalculateSizeOfObjectFromFlags() const {
  DCHECK_GE(state(), kStateFrozen);

  if (has_xstate_data_ || (misc_info_.Flags1 & MINIDUMP_MISC5_PROCESS_COOKIE)) {
    return sizeof(MINIDUMP_MISC_INFO_5);
  }
  if (misc_info_.Flags1 & MINIDUMP_MISC4_BUILDSTRING) {
    return sizeof(MINIDUMP_MISC_INFO_4);
  }
  if (misc_info_.Flags1 &
      (MINIDUMP_MISC3_PROCESS_INTEGRITY | MINIDUMP_MISC3_PROCESS_EXECUTE_FLAGS |
       MINIDUMP_MISC3_TIMEZONE | MINIDUMP_MISC3_PROTECTED_PROCESS)) {
    return sizeof(MINIDUMP_MISC_INFO_3);
  }
  if (misc_info_.Flags1 & MINIDUMP_MISC1_PROCESSOR_POWER_INFO) {
    return sizeof(MINIDUMP_MISC_INFO_2);
  }
  return sizeof(MINIDUMP_MISC_INFO);
}

}  // namespace crashpad
