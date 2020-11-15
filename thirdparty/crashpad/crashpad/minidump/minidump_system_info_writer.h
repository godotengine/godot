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

#ifndef CRASHPAD_MINIDUMP_MINIDUMP_SYSTEM_INFO_WRITER_H_
#define CRASHPAD_MINIDUMP_MINIDUMP_SYSTEM_INFO_WRITER_H_

#include <windows.h>
#include <dbghelp.h>
#include <stdint.h>
#include <sys/types.h>

#include <memory>
#include <string>
#include <vector>

#include "base/macros.h"
#include "minidump/minidump_extensions.h"
#include "minidump/minidump_stream_writer.h"
#include "minidump/minidump_writable.h"

namespace crashpad {

class SystemSnapshot;

namespace internal {
class MinidumpUTF16StringWriter;
}  // namespace internal

//! \brief The writer for a MINIDUMP_SYSTEM_INFO stream in a minidump file.
class MinidumpSystemInfoWriter final : public internal::MinidumpStreamWriter {
 public:
  MinidumpSystemInfoWriter();
  ~MinidumpSystemInfoWriter() override;

  //! \brief Initializes MINIDUMP_SYSTEM_INFO based on \a system_snapshot.
  //!
  //! \param[in] system_snapshot The system snapshot to use as source data.
  //!
  //! \note Valid in #kStateMutable. No mutator methods may be called before
  //!     this method, and it is not normally necessary to call any mutator
  //!     methods after this method.
  void InitializeFromSnapshot(const SystemSnapshot* system_snapshot);

  //! \brief Sets MINIDUMP_SYSTEM_INFO::ProcessorArchitecture.
  void SetCPUArchitecture(MinidumpCPUArchitecture processor_architecture) {
    system_info_.ProcessorArchitecture = processor_architecture;
  }

  //! \brief Sets MINIDUMP_SYSTEM_INFO::ProcessorLevel and
  //!     MINIDUMP_SYSTEM_INFO::ProcessorRevision.
  void SetCPULevelAndRevision(uint16_t processor_level,
                              uint16_t processor_revision) {
    system_info_.ProcessorLevel = processor_level;
    system_info_.ProcessorRevision = processor_revision;
  }

  //! \brief Sets MINIDUMP_SYSTEM_INFO::NumberOfProcessors.
  void SetCPUCount(uint8_t number_of_processors) {
    system_info_.NumberOfProcessors = number_of_processors;
  }

  //! \brief Sets MINIDUMP_SYSTEM_INFO::PlatformId.
  void SetOS(MinidumpOS platform_id) { system_info_.PlatformId = platform_id; }

  //! \brief Sets MINIDUMP_SYSTEM_INFO::ProductType.
  void SetOSType(MinidumpOSType product_type) {
    system_info_.ProductType = product_type;
  }

  //! \brief Sets MINIDUMP_SYSTEM_INFO::MajorVersion,
  //!     MINIDUMP_SYSTEM_INFO::MinorVersion, and
  //!     MINIDUMP_SYSTEM_INFO::BuildNumber.
  void SetOSVersion(uint32_t major_version,
                    uint32_t minor_version,
                    uint32_t build_number) {
    system_info_.MajorVersion = major_version;
    system_info_.MinorVersion = minor_version;
    system_info_.BuildNumber = build_number;
  }

  //! \brief Arranges for MINIDUMP_SYSTEM_INFO::CSDVersionRva to point to a
  //!     MINIDUMP_STRING containing the supplied string.
  //!
  //! This method must be called prior to Freeze(). A CSD version is required
  //! in all MINIDUMP_SYSTEM_INFO streams. An empty string is an acceptable
  //! value.
  void SetCSDVersion(const std::string& csd_version);

  //! \brief Sets MINIDUMP_SYSTEM_INFO::SuiteMask.
  void SetSuiteMask(uint16_t suite_mask) {
    system_info_.SuiteMask = suite_mask;
  }

  //! \brief Sets \ref CPU_INFORMATION::VendorId
  //!     "MINIDUMP_SYSTEM_INFO::Cpu::X86CpuInfo::VendorId".
  //!
  //! This is only valid if SetCPUArchitecture() has been used to set the CPU
  //! architecture to #kMinidumpCPUArchitectureX86 or
  //! #kMinidumpCPUArchitectureX86Win64.
  //!
  //! \param[in] ebx The first 4 bytes of the CPU vendor string, the value
  //!     reported in `cpuid 0` `ebx`.
  //! \param[in] edx The middle 4 bytes of the CPU vendor string, the value
  //!     reported in `cpuid 0` `edx`.
  //! \param[in] ecx The last 4 bytes of the CPU vendor string, the value
  //!     reported by `cpuid 0` `ecx`.
  //!
  //! \note Do not call this method if SetCPUArchitecture() has been used to set
  //!     the CPU architecture to #kMinidumpCPUArchitectureAMD64.
  //!
  //! \sa SetCPUX86VendorString()
  void SetCPUX86Vendor(uint32_t ebx, uint32_t edx, uint32_t ecx);

  //! \brief Sets \ref CPU_INFORMATION::VendorId
  //!     "MINIDUMP_SYSTEM_INFO::Cpu::X86CpuInfo::VendorId".
  //!
  //! This is only valid if SetCPUArchitecture() has been used to set the CPU
  //! architecture to #kMinidumpCPUArchitectureX86 or
  //! #kMinidumpCPUArchitectureX86Win64.
  //!
  //! \param[in] vendor The entire CPU vendor string, which must be exactly 12
  //!     bytes long.
  //!
  //! \note Do not call this method if SetCPUArchitecture() has been used to set
  //!     the CPU architecture to #kMinidumpCPUArchitectureAMD64.
  //!
  //! \sa SetCPUX86Vendor()
  void SetCPUX86VendorString(const std::string& vendor);

  //! \brief Sets \ref CPU_INFORMATION::VersionInformation
  //!     "MINIDUMP_SYSTEM_INFO::Cpu::X86CpuInfo::VersionInformation" and
  //!     \ref CPU_INFORMATION::FeatureInformation
  //!     "MINIDUMP_SYSTEM_INFO::Cpu::X86CpuInfo::FeatureInformation".
  //!
  //! This is only valid if SetCPUArchitecture() has been used to set the CPU
  //! architecture to #kMinidumpCPUArchitectureX86 or
  //! #kMinidumpCPUArchitectureX86Win64.
  //!
  //! \note Do not call this method if SetCPUArchitecture() has been used to set
  //!     the CPU architecture to #kMinidumpCPUArchitectureAMD64.
  void SetCPUX86VersionAndFeatures(uint32_t version, uint32_t features);

  //! \brief Sets \ref CPU_INFORMATION::AMDExtendedCpuFeatures
  //!     "MINIDUMP_SYSTEM_INFO::Cpu::X86CpuInfo::AMDExtendedCPUFeatures".
  //!
  //! This is only valid if SetCPUArchitecture() has been used to set the CPU
  //! architecture to #kMinidumpCPUArchitectureX86 or
  //! #kMinidumpCPUArchitectureX86Win64, and if SetCPUX86Vendor() or
  //! SetCPUX86VendorString() has been used to set the CPU vendor to
  //! “AuthenticAMD”.
  //!
  //! \note Do not call this method if SetCPUArchitecture() has been used to set
  //!     the CPU architecture to #kMinidumpCPUArchitectureAMD64.
  void SetCPUX86AMDExtendedFeatures(uint32_t extended_features);

  //! \brief Sets \ref CPU_INFORMATION::ProcessorFeatures
  //!     "MINIDUMP_SYSTEM_INFO::Cpu::OtherCpuInfo::ProcessorFeatures".
  //!
  //! This is only valid if SetCPUArchitecture() has been used to set the CPU
  //! architecture to an architecture other than #kMinidumpCPUArchitectureX86
  //! or #kMinidumpCPUArchitectureX86Win64.
  //!
  //! \note This method may be called if SetCPUArchitecture() has been used to
  //!     set the CPU architecture to #kMinidumpCPUArchitectureAMD64.
  void SetCPUOtherFeatures(uint64_t features_0, uint64_t features_1);

 protected:
  // MinidumpWritable:
  bool Freeze() override;
  size_t SizeOfObject() override;
  std::vector<MinidumpWritable*> Children() override;
  bool WriteObject(FileWriterInterface* file_writer) override;

  // MinidumpStreamWriter:
  MinidumpStreamType StreamType() const override;

 private:
  MINIDUMP_SYSTEM_INFO system_info_;
  std::unique_ptr<internal::MinidumpUTF16StringWriter> csd_version_;

  DISALLOW_COPY_AND_ASSIGN(MinidumpSystemInfoWriter);
};

}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_MINIDUMP_SYSTEM_INFO_WRITER_H_
