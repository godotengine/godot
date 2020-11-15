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

#ifndef CRASHPAD_SNAPSHOT_SYSTEM_SNAPSHOT_H_
#define CRASHPAD_SNAPSHOT_SYSTEM_SNAPSHOT_H_

#include <stdint.h>
#include <sys/types.h>

#include <string>

#include "snapshot/cpu_architecture.h"

namespace crashpad {

//! \brief An abstract interface to a snapshot representing the state of a
//!     system, comprising an operating system, CPU architecture, and various
//!     other characteristics.
class SystemSnapshot {
 public:
  virtual ~SystemSnapshot() {}

  //! \brief A system’s operating system family.
  enum OperatingSystem {
    //! \brief The snapshot system’s operating system is unknown.
    kOperatingSystemUnknown = 0,

    //! \brief macOS.
    kOperatingSystemMacOSX,

    //! \brief Windows.
    kOperatingSystemWindows,

    //! \brief Linux.
    kOperatingSystemLinux,

    //! \brief Android.
    kOperatingSystemAndroid,

    //! \brief Fuchsia.
    kOperatingSystemFuchsia,
  };

  //! \brief A system’s daylight saving time status.
  //!
  //! The daylight saving time status is taken partially from the system’s
  //! locale configuration. This determines whether daylight saving time is
  //! ever observed on the system. If it is, the snapshot’s time
  //! (ProcessSnapshot::SnapshotTime()) is used to determine whether the system
  //! was observing daylight saving time at the time of the snapshot.
  enum DaylightSavingTimeStatus {
    //! \brief Daylight saving time is never observed on the snapshot system.
    kDoesNotObserveDaylightSavingTime = 0,

    //! \brief Daylight saving time is observed on the snapshot system when in
    //!     effect, but standard time was in effect at the time of the snapshot.
    kObservingStandardTime,

    //! \brief Daylight saving time is observed on the snapshot system when in
    //!     effect, and daylight saving time was in effect at the time of the
    //!     snapshot.
    kObservingDaylightSavingTime,
  };

  //! \brief Returns the snapshot system’s CPU architecture.
  //!
  //! In some cases, a system may be able to run processes of multiple specific
  //! architecture types. For example, systems based on 64-bit architectures
  //! such as x86_64 are often able to run 32-bit code of another architecture
  //! in the same family, such as 32-bit x86. On these systems, this method will
  //! return the architecture of the process that the snapshot is associated
  //! with, provided that the SystemSnapshot object was obtained from
  //! ProcessSnapshot::System(). This renders one aspect of this method’s return
  //! value a process attribute rather than a system attribute, but it’s defined
  //! here rather than in ProcessSnapshot because the CPU architecture is a
  //! better conceptual fit for the system abstraction alongside these other
  //! related methods.
  virtual CPUArchitecture GetCPUArchitecture() const = 0;

  //! \brief Returns the snapshot system’s CPU revision.
  //!
  //! For x86-family CPUs (including x86_64 and 32-bit x86), this is the CPU
  //! family, model, and stepping ID values from `cpuid 1` `eax`. The family and
  //! model values are adjusted to take the extended family and model IDs into
  //! account. These values are encoded in this method’s return value with the
  //! family in the high high 16 bits, the model in the next 8 bits, and the
  //! stepping in the low 8 bits.
  //!
  //! \return A CPU architecture-specific value identifying the CPU revision.
  virtual uint32_t CPURevision() const = 0;

  //! \brief Returns the total number of CPUs present in the snapshot system.
  virtual uint8_t CPUCount() const = 0;

  //! \brief Returns the vendor of the snapshot system’s CPUs.
  //!
  //! For x86-family CPUs (including x86_64 and 32-bit x86), this is the CPU
  //! vendor identification string as encoded in `cpuid 0` `ebx`, `edx`, and
  //! `ecx`.
  //!
  //! \return A string identifying the vendor of the snapshot system’s CPUs.
  virtual std::string CPUVendor() const = 0;

  //! \brief Returns frequency information about the snapshot system’s CPUs in
  //!     \a current_hz and \a max_hz.
  //!
  //! \param[out] current_hz The snapshot system’s CPU clock frequency in Hz at
  //!     the time of the snapshot.
  //! \param[out] max_hz The snapshot system’s maximum possible CPU clock
  //!     frequency.
  virtual void CPUFrequency(uint64_t* current_hz, uint64_t* max_hz) const = 0;

  //! \brief Returns an x86-family snapshot system’s CPU signature.
  //!
  //! This is the family, model, and stepping ID values as encoded in `cpuid 1`
  //! `eax`.
  //!
  //! This method must only be called when GetCPUArchitecture() indicates an
  //! x86-family CPU architecture (#kCPUArchitectureX86 or
  //! #kCPUArchitectureX86_64).
  //!
  //! \return An x86 family-specific value identifying the CPU signature.
  virtual uint32_t CPUX86Signature() const = 0;

  //! \brief Returns an x86-family snapshot system’s CPU features.
  //!
  //! This is the feature information as encoded in `cpuid 1` `edx` and `ecx`.
  //! `edx` is placed in the low half of the return value, and `ecx` is placed
  //! in the high half.
  //!
  //! This method must only be called when GetCPUArchitecture() indicates an
  //! x86-family CPU architecture (#kCPUArchitectureX86 or
  //! #kCPUArchitectureX86_64).
  //!
  //! \return An x86 family-specific value identifying CPU features.
  //!
  //! \sa CPUX86ExtendedFeatures()
  //! \sa CPUX86Leaf7Features()
  virtual uint64_t CPUX86Features() const = 0;

  //! \brief Returns an x86-family snapshot system’s extended CPU features.
  //!
  //! This is the extended feature information as encoded in `cpuid 0x80000001`
  //! `edx` and `ecx`. `edx` is placed in the low half of the return value, and
  //! `ecx` is placed in the high half.
  //!
  //! This method must only be called when GetCPUArchitecture() indicates an
  //! x86-family CPU architecture (#kCPUArchitectureX86 or
  //! #kCPUArchitectureX86_64).
  //!
  //! \return An x86 family-specific value identifying extended CPU features.
  //!
  //! \sa CPUX86Features()
  //! \sa CPUX86Leaf7Features()
  virtual uint64_t CPUX86ExtendedFeatures() const = 0;

  //! \brief Returns an x86-family snapshot system’s “leaf 7” CPU features.
  //!
  //! This is the “leaf 7” feature information as encoded in `cpuid 7` `ebx`. If
  //! `cpuid 7` is not supported by the snapshot CPU, this returns `0`.
  //!
  //! This method must only be called when GetCPUArchitecture() indicates an
  //! x86-family CPU architecture (#kCPUArchitectureX86 or
  //! #kCPUArchitectureX86_64).
  //!
  //! \return An x86 family-specific value identifying “leaf 7” CPU features.
  //!
  //! \sa CPUX86Features()
  //! \sa CPUX86ExtendedFeatures()
  virtual uint32_t CPUX86Leaf7Features() const = 0;

  //! \brief Returns an x86-family snapshot system’s CPU’s support for the SSE
  //!     DAZ (“denormals are zeros”) mode.
  //!
  //! This determines whether the CPU supports DAZ mode at all, not whether this
  //! mode is enabled for any particular thread. DAZ mode support is detected by
  //! examining the DAZ bit in the `mxcsr_mask` field of the floating-point
  //! context saved by `fxsave`.
  //!
  //! This method must only be called when GetCPUArchitecture() indicates an
  //! x86-family CPU architecture (#kCPUArchitectureX86 or
  //! #kCPUArchitectureX86_64).
  //!
  //! \return `true` if the snapshot system’s CPUs support the SSE DAZ mode,
  //!     `false` if they do not.
  virtual bool CPUX86SupportsDAZ() const = 0;

  //! \brief Returns the snapshot system’s operating system family.
  virtual OperatingSystem GetOperatingSystem() const = 0;

  //! \brief Returns whether the snapshot system runs a server variant of its
  //!     operating system.
  virtual bool OSServer() const = 0;

  //! \brief Returns the snapshot system’s operating system version information
  //!     in \a major, \a minor, \a bugfix, and \a build.
  //!
  //! \param[out] major The snapshot system’s operating system’s first (major)
  //!     version number component. This would be `10` for macOS 10.12.1, and
  //!     `6` for Windows 7 (NT 6.1) SP1 version 6.1.7601.
  //! \param[out] minor The snapshot system’s operating system’s second (minor)
  //!     version number component. This would be `12` for macOS 10.12.1, and
  //!     `1` for Windows 7 (NT 6.1) SP1 version 6.1.7601.
  //! \param[out] bugfix The snapshot system’s operating system’s third (bugfix)
  //!     version number component. This would be `1` for macOS 10.12.1, and
  //!     `7601` for Windows 7 (NT 6.1) SP1 version 6.1.7601.
  //! \param[out] build A string further identifying an operating system
  //!     version. For macOS 10.12.1, this would be `"16B2657"`. For Windows,
  //!     this would be `"Service Pack 1"` if that service pack was installed.
  //!     On Android, the `ro.build.fingerprint` system property would be
  //!     appended. For Linux and other Unix-like systems, this would be the
  //!     kernel version from `uname -srvm`, possibly with additional
  //!     information appended.
  virtual void OSVersion(int* major,
                         int* minor,
                         int* bugfix,
                         std::string* build) const = 0;

  //! \brief Returns the snapshot system’s full operating system version
  //!     information in string format.
  //!
  //! For macOS, the string contains values from the operating system and
  //! kernel. A macOS 10.12.1 system snapshot would be identified as `"Mac OS
  //! X 10.12.1 (16B2657); Darwin 16.1.0 Darwin Kernel Version 16.1.0: Wed Oct
  //! 19 20:31:56 PDT 2016; root:xnu-3789.21.4~4/RELEASE_X86_64 x86_64"`.
  virtual std::string OSVersionFull() const = 0;

  //! \brief Returns a description of the snapshot system’s hardware in string
  //!     format.
  //!
  //! For macOS, the string contains the Mac model and board ID. A mid-2014 15"
  //! MacBook Pro would be identified as `"MacBookPro11,3
  //! (Mac-2BD1B31983FE1663)"`.
  virtual std::string MachineDescription() const = 0;

  //! \brief Returns the status of the NX (no-execute, or XD, execute-disable)
  //!     feature on the snapshot system.
  //!
  //! This refers to a feature that allows mapped readable pages to be marked
  //! as non-executable.
  //!
  //! \return `true` if the snapshot system supports NX and it is enabled.
  virtual bool NXEnabled() const = 0;

  //! \brief Returns time zone information from the snapshot system, based on
  //!     its locale configuration and real-time clock.
  //!
  //! \param[out] dst_status Whether the location observes daylight saving time,
  //!     and if so, whether it or standard time is currently being observed.
  //! \param[out] standard_offset_seconds The number of seconds that the
  //!     location’s time zone is east (ahead) of UTC during standard time.
  //! \param[out] daylight_offset_seconds The number of seconds that the
  //!     location’s time zone is east (ahead) of UTC during daylight saving.
  //!     time.
  //! \param[out] standard_name The name of the time zone while standard time is
  //!     being observed.
  //! \param[out] daylight_name The name of the time zone while daylight saving
  //!     time is being observed.
  virtual void TimeZone(DaylightSavingTimeStatus* dst_status,
                        int* standard_offset_seconds,
                        int* daylight_offset_seconds,
                        std::string* standard_name,
                        std::string* daylight_name) const = 0;
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_SYSTEM_SNAPSHOT_H_
