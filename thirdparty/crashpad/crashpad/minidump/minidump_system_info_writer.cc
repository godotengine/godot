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

#include "minidump/minidump_system_info_writer.h"

#include <string.h>

#include "base/logging.h"
#include "minidump/minidump_string_writer.h"
#include "snapshot/system_snapshot.h"
#include "util/file/file_writer.h"
#include "util/misc/arraysize_unsafe.h"
#include "util/misc/implicit_cast.h"

namespace crashpad {

namespace {

uint64_t AMD64FeaturesFromSystemSnapshot(
    const SystemSnapshot* system_snapshot) {
#define ADD_FEATURE(minidump_bit) (UINT64_C(1) << (minidump_bit))

  // Features for which no cpuid bits are present, but that always exist on
  // x86_64. cmpxchg is supported on 486 and later.
  uint64_t minidump_features = ADD_FEATURE(PF_COMPARE_EXCHANGE_DOUBLE);

#define MAP_FEATURE(features, cpuid_bit, minidump_bit)                        \
  do {                                                                        \
    if ((features) & (implicit_cast<decltype(features)>(1) << (cpuid_bit))) { \
      minidump_features |= ADD_FEATURE(minidump_bit);                         \
    }                                                                         \
  } while (false)

#define F_TSC 4
#define F_PAE 6
#define F_MMX 23
#define F_SSE 25
#define F_SSE2 26
#define F_SSE3 32
#define F_CX16 45
#define F_XSAVE 58
#define F_RDRAND 62

  uint64_t cpuid_features = system_snapshot->CPUX86Features();

  MAP_FEATURE(cpuid_features, F_TSC, PF_RDTSC_INSTRUCTION_AVAILABLE);
  MAP_FEATURE(cpuid_features, F_PAE, PF_PAE_ENABLED);
  MAP_FEATURE(cpuid_features, F_MMX, PF_MMX_INSTRUCTIONS_AVAILABLE);
  MAP_FEATURE(cpuid_features, F_SSE, PF_XMMI_INSTRUCTIONS_AVAILABLE);
  MAP_FEATURE(cpuid_features, F_SSE2, PF_XMMI64_INSTRUCTIONS_AVAILABLE);
  MAP_FEATURE(cpuid_features, F_SSE3, PF_SSE3_INSTRUCTIONS_AVAILABLE);
  MAP_FEATURE(cpuid_features, F_CX16, PF_COMPARE_EXCHANGE128);
  MAP_FEATURE(cpuid_features, F_XSAVE, PF_XSAVE_ENABLED);
  MAP_FEATURE(cpuid_features, F_RDRAND, PF_RDRAND_INSTRUCTION_AVAILABLE);

#define FX_XD 20
#define FX_RDTSCP 27
#define FX_3DNOW 31

  uint64_t extended_features = system_snapshot->CPUX86ExtendedFeatures();

  MAP_FEATURE(extended_features, FX_RDTSCP, PF_RDTSCP_INSTRUCTION_AVAILABLE);
  MAP_FEATURE(extended_features, FX_3DNOW, PF_3DNOW_INSTRUCTIONS_AVAILABLE);

#define F7_FSGSBASE 0

  uint32_t leaf7_features = system_snapshot->CPUX86Leaf7Features();

  MAP_FEATURE(leaf7_features, F7_FSGSBASE, PF_RDWRFSGSBASE_AVAILABLE);

  // This feature bit should be set if NX (XD, DEP) is enabled, not just if it’s
  // available on the CPU as indicated by the FX_XD bit.
  if (system_snapshot->NXEnabled()) {
    minidump_features |= ADD_FEATURE(PF_NX_ENABLED);
  }

  if (system_snapshot->CPUX86SupportsDAZ()) {
    minidump_features |= ADD_FEATURE(PF_SSE_DAZ_MODE_AVAILABLE);
  }

  // PF_SECOND_LEVEL_ADDRESS_TRANSLATION can’t be determined without consulting
  // model-specific registers, a privileged operation. The exact use of
  // PF_VIRT_FIRMWARE_ENABLED is unknown. PF_FASTFAIL_AVAILABLE is irrelevant
  // outside of Windows.

#undef MAP_FEATURE
#undef ADD_FEATURE

  return minidump_features;
}

}  // namespace

MinidumpSystemInfoWriter::MinidumpSystemInfoWriter()
    : MinidumpStreamWriter(), system_info_(), csd_version_() {
  system_info_.ProcessorArchitecture = kMinidumpCPUArchitectureUnknown;
}

MinidumpSystemInfoWriter::~MinidumpSystemInfoWriter() {
}

void MinidumpSystemInfoWriter::InitializeFromSnapshot(
    const SystemSnapshot* system_snapshot) {
  DCHECK_EQ(state(), kStateMutable);
  DCHECK(!csd_version_);

  MinidumpCPUArchitecture cpu_architecture;
  switch (system_snapshot->GetCPUArchitecture()) {
    case kCPUArchitectureX86:
      cpu_architecture = kMinidumpCPUArchitectureX86;
      break;
    case kCPUArchitectureX86_64:
      cpu_architecture = kMinidumpCPUArchitectureAMD64;
      break;
    case kCPUArchitectureARM:
      cpu_architecture = kMinidumpCPUArchitectureARM;
      break;
    case kCPUArchitectureARM64:
      cpu_architecture = kMinidumpCPUArchitectureARM64;
      break;
    default:
      NOTREACHED();
      cpu_architecture = kMinidumpCPUArchitectureUnknown;
      break;
  }
  SetCPUArchitecture(cpu_architecture);

  uint32_t cpu_revision = system_snapshot->CPURevision();
  SetCPULevelAndRevision((cpu_revision & 0xffff0000) >> 16,
                         cpu_revision & 0x0000ffff);
  SetCPUCount(system_snapshot->CPUCount());

  if (cpu_architecture == kMinidumpCPUArchitectureX86) {
    std::string cpu_vendor = system_snapshot->CPUVendor();
    SetCPUX86VendorString(cpu_vendor);

    // The minidump file format only has room for the bottom 32 bits of CPU
    // features and extended CPU features.
    SetCPUX86VersionAndFeatures(system_snapshot->CPUX86Signature(),
                                system_snapshot->CPUX86Features() & 0xffffffff);

    if (cpu_vendor == "AuthenticAMD") {
      SetCPUX86AMDExtendedFeatures(
          system_snapshot->CPUX86ExtendedFeatures() & 0xffffffff);
    }
  } else if (cpu_architecture == kMinidumpCPUArchitectureAMD64) {
    SetCPUOtherFeatures(AMD64FeaturesFromSystemSnapshot(system_snapshot), 0);
  }

  MinidumpOS operating_system;
  switch (system_snapshot->GetOperatingSystem()) {
    case SystemSnapshot::kOperatingSystemMacOSX:
      operating_system = kMinidumpOSMacOSX;
      break;
    case SystemSnapshot::kOperatingSystemWindows:
      operating_system = kMinidumpOSWin32NT;
      break;
    case SystemSnapshot::kOperatingSystemLinux:
      operating_system = kMinidumpOSLinux;
      break;
    case SystemSnapshot::kOperatingSystemAndroid:
      operating_system = kMinidumpOSAndroid;
      break;
    case SystemSnapshot::kOperatingSystemFuchsia:
      operating_system = kMinidumpOSFuchsia;
      break;
    default:
      NOTREACHED();
      operating_system = kMinidumpOSUnknown;
      break;
  }
  SetOS(operating_system);

  SetOSType(system_snapshot->OSServer() ? kMinidumpOSTypeServer
                                        : kMinidumpOSTypeWorkstation);

  int major;
  int minor;
  int bugfix;
  std::string build;
  system_snapshot->OSVersion(&major, &minor, &bugfix, &build);
  SetOSVersion(major, minor, bugfix);
  SetCSDVersion(build);
}

void MinidumpSystemInfoWriter::SetCSDVersion(const std::string& csd_version) {
  DCHECK_EQ(state(), kStateMutable);

  if (!csd_version_) {
    csd_version_.reset(new internal::MinidumpUTF16StringWriter());
  }

  csd_version_->SetUTF8(csd_version);
}

void MinidumpSystemInfoWriter::SetCPUX86Vendor(uint32_t ebx,
                                               uint32_t edx,
                                               uint32_t ecx) {
  DCHECK_EQ(state(), kStateMutable);
  DCHECK(system_info_.ProcessorArchitecture == kMinidumpCPUArchitectureX86 ||
         system_info_.ProcessorArchitecture ==
             kMinidumpCPUArchitectureX86Win64);

  static_assert(ARRAYSIZE_UNSAFE(system_info_.Cpu.X86CpuInfo.VendorId) == 3,
                "VendorId must have 3 elements");

  system_info_.Cpu.X86CpuInfo.VendorId[0] = ebx;
  system_info_.Cpu.X86CpuInfo.VendorId[1] = edx;
  system_info_.Cpu.X86CpuInfo.VendorId[2] = ecx;
}

void MinidumpSystemInfoWriter::SetCPUX86VendorString(
    const std::string& vendor) {
  DCHECK_EQ(state(), kStateMutable);
  CHECK_EQ(vendor.size(), sizeof(system_info_.Cpu.X86CpuInfo.VendorId));

  uint32_t registers[3];
  static_assert(
      sizeof(registers) == sizeof(system_info_.Cpu.X86CpuInfo.VendorId),
      "VendorId sizes must be equal");

  for (size_t index = 0; index < arraysize(registers); ++index) {
    memcpy(&registers[index],
           &vendor[index * sizeof(*registers)],
           sizeof(*registers));
  }

  SetCPUX86Vendor(registers[0], registers[1], registers[2]);
}

void MinidumpSystemInfoWriter::SetCPUX86VersionAndFeatures(uint32_t version,
                                                           uint32_t features) {
  DCHECK_EQ(state(), kStateMutable);
  DCHECK(system_info_.ProcessorArchitecture == kMinidumpCPUArchitectureX86 ||
         system_info_.ProcessorArchitecture ==
             kMinidumpCPUArchitectureX86Win64);

  system_info_.Cpu.X86CpuInfo.VersionInformation = version;
  system_info_.Cpu.X86CpuInfo.FeatureInformation = features;
}

void MinidumpSystemInfoWriter::SetCPUX86AMDExtendedFeatures(
    uint32_t extended_features) {
  DCHECK_EQ(state(), kStateMutable);
  DCHECK(system_info_.ProcessorArchitecture == kMinidumpCPUArchitectureX86 ||
         system_info_.ProcessorArchitecture ==
             kMinidumpCPUArchitectureX86Win64);
  DCHECK(system_info_.Cpu.X86CpuInfo.VendorId[0] == 'htuA' &&
         system_info_.Cpu.X86CpuInfo.VendorId[1] == 'itne' &&
         system_info_.Cpu.X86CpuInfo.VendorId[2] == 'DMAc');

  system_info_.Cpu.X86CpuInfo.AMDExtendedCpuFeatures = extended_features;
}

void MinidumpSystemInfoWriter::SetCPUOtherFeatures(uint64_t features_0,
                                                   uint64_t features_1) {
  DCHECK_EQ(state(), kStateMutable);
  DCHECK(system_info_.ProcessorArchitecture != kMinidumpCPUArchitectureX86 &&
         system_info_.ProcessorArchitecture !=
             kMinidumpCPUArchitectureX86Win64);

  static_assert(
      ARRAYSIZE_UNSAFE(system_info_.Cpu.OtherCpuInfo.ProcessorFeatures) == 2,
      "ProcessorFeatures must have 2 elements");

  system_info_.Cpu.OtherCpuInfo.ProcessorFeatures[0] = features_0;
  system_info_.Cpu.OtherCpuInfo.ProcessorFeatures[1] = features_1;
}

bool MinidumpSystemInfoWriter::Freeze() {
  DCHECK_EQ(state(), kStateMutable);
  CHECK(csd_version_);

  if (!MinidumpStreamWriter::Freeze()) {
    return false;
  }

  csd_version_->RegisterRVA(&system_info_.CSDVersionRva);

  return true;
}

size_t MinidumpSystemInfoWriter::SizeOfObject() {
  DCHECK_GE(state(), kStateFrozen);

  return sizeof(system_info_);
}

std::vector<internal::MinidumpWritable*> MinidumpSystemInfoWriter::Children() {
  DCHECK_GE(state(), kStateFrozen);
  DCHECK(csd_version_);

  std::vector<MinidumpWritable*> children(1, csd_version_.get());
  return children;
}

bool MinidumpSystemInfoWriter::WriteObject(FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);

  return file_writer->Write(&system_info_, sizeof(system_info_));
}

MinidumpStreamType MinidumpSystemInfoWriter::StreamType() const {
  return kMinidumpStreamTypeSystemInfo;
}

}  // namespace crashpad
