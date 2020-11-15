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

#include <algorithm>
#include <string>
#include <utility>

#include "base/compiler_specific.h"
#include "gtest/gtest.h"
#include "minidump/minidump_file_writer.h"
#include "minidump/test/minidump_file_writer_test_util.h"
#include "minidump/test/minidump_string_writer_test_util.h"
#include "minidump/test/minidump_writable_test_util.h"
#include "snapshot/test/test_system_snapshot.h"
#include "test/gtest_death.h"
#include "util/file/string_file.h"

namespace crashpad {
namespace test {
namespace {

void GetSystemInfoStream(const std::string& file_contents,
                         size_t csd_version_length,
                         const MINIDUMP_SYSTEM_INFO** system_info,
                         const MINIDUMP_STRING** csd_version) {
  // The expected number of bytes for the CSD version’s MINIDUMP_STRING::Buffer.
  MINIDUMP_STRING* tmp;
  ALLOW_UNUSED_LOCAL(tmp);
  const size_t kCSDVersionBytes = csd_version_length * sizeof(tmp->Buffer[0]);
  const size_t kCSDVersionBytesWithNUL =
      kCSDVersionBytes + sizeof(tmp->Buffer[0]);

  constexpr size_t kDirectoryOffset = sizeof(MINIDUMP_HEADER);
  constexpr size_t kSystemInfoStreamOffset =
      kDirectoryOffset + sizeof(MINIDUMP_DIRECTORY);
  constexpr size_t kCSDVersionOffset =
      kSystemInfoStreamOffset + sizeof(MINIDUMP_SYSTEM_INFO);
  const size_t kFileSize =
      kCSDVersionOffset + sizeof(MINIDUMP_STRING) + kCSDVersionBytesWithNUL;

  ASSERT_EQ(file_contents.size(), kFileSize);

  const MINIDUMP_DIRECTORY* directory;
  const MINIDUMP_HEADER* header =
      MinidumpHeaderAtStart(file_contents, &directory);
  ASSERT_NO_FATAL_FAILURE(VerifyMinidumpHeader(header, 1, 0));
  ASSERT_TRUE(directory);

  ASSERT_EQ(directory[0].StreamType, kMinidumpStreamTypeSystemInfo);
  EXPECT_EQ(directory[0].Location.Rva, kSystemInfoStreamOffset);

  *system_info = MinidumpWritableAtLocationDescriptor<MINIDUMP_SYSTEM_INFO>(
      file_contents, directory[0].Location);
  ASSERT_TRUE(system_info);

  EXPECT_EQ((*system_info)->CSDVersionRva, kCSDVersionOffset);

  *csd_version =
      MinidumpStringAtRVA(file_contents, (*system_info)->CSDVersionRva);
  EXPECT_EQ((*csd_version)->Length, kCSDVersionBytes);
}

TEST(MinidumpSystemInfoWriter, Empty) {
  MinidumpFileWriter minidump_file_writer;
  auto system_info_writer = std::make_unique<MinidumpSystemInfoWriter>();

  system_info_writer->SetCSDVersion(std::string());

  ASSERT_TRUE(minidump_file_writer.AddStream(std::move(system_info_writer)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  const MINIDUMP_SYSTEM_INFO* system_info = nullptr;
  const MINIDUMP_STRING* csd_version = nullptr;

  ASSERT_NO_FATAL_FAILURE(
      GetSystemInfoStream(string_file.string(), 0, &system_info, &csd_version));

  EXPECT_EQ(system_info->ProcessorArchitecture,
            kMinidumpCPUArchitectureUnknown);
  EXPECT_EQ(system_info->ProcessorLevel, 0u);
  EXPECT_EQ(system_info->ProcessorRevision, 0u);
  EXPECT_EQ(system_info->NumberOfProcessors, 0u);
  EXPECT_EQ(system_info->ProductType, 0u);
  EXPECT_EQ(system_info->MajorVersion, 0u);
  EXPECT_EQ(system_info->MinorVersion, 0u);
  EXPECT_EQ(system_info->BuildNumber, 0u);
  EXPECT_EQ(system_info->PlatformId, 0u);
  EXPECT_EQ(system_info->SuiteMask, 0u);
  EXPECT_EQ(system_info->Cpu.X86CpuInfo.VendorId[0], 0u);
  EXPECT_EQ(system_info->Cpu.X86CpuInfo.VendorId[1], 0u);
  EXPECT_EQ(system_info->Cpu.X86CpuInfo.VendorId[2], 0u);
  EXPECT_EQ(system_info->Cpu.X86CpuInfo.VersionInformation, 0u);
  EXPECT_EQ(system_info->Cpu.X86CpuInfo.FeatureInformation, 0u);
  EXPECT_EQ(system_info->Cpu.X86CpuInfo.AMDExtendedCpuFeatures, 0u);
  EXPECT_EQ(system_info->Cpu.OtherCpuInfo.ProcessorFeatures[0], 0u);
  EXPECT_EQ(system_info->Cpu.OtherCpuInfo.ProcessorFeatures[1], 0u);

  EXPECT_EQ(csd_version->Buffer[0], '\0');
}

TEST(MinidumpSystemInfoWriter, X86_Win) {
  MinidumpFileWriter minidump_file_writer;
  auto system_info_writer = std::make_unique<MinidumpSystemInfoWriter>();

  constexpr MinidumpCPUArchitecture kCPUArchitecture =
      kMinidumpCPUArchitectureX86;
  constexpr uint16_t kCPULevel = 0x0010;
  constexpr uint16_t kCPURevision = 0x0602;
  constexpr uint8_t kCPUCount = 1;
  constexpr MinidumpOS kOS = kMinidumpOSWin32NT;
  constexpr MinidumpOSType kOSType = kMinidumpOSTypeWorkstation;
  constexpr uint32_t kOSVersionMajor = 6;
  constexpr uint32_t kOSVersionMinor = 1;
  constexpr uint32_t kOSVersionBuild = 7601;
  static constexpr char kCSDVersion[] = "Service Pack 1";
  constexpr uint16_t kSuiteMask = VER_SUITE_SINGLEUSERTS;
  static constexpr char kCPUVendor[] = "AuthenticAMD";
  constexpr uint32_t kCPUVersion = 0x00100f62;
  constexpr uint32_t kCPUFeatures = 0x078bfbff;
  constexpr uint32_t kAMDFeatures = 0xefd3fbff;

  uint32_t cpu_vendor_registers[3];
  ASSERT_EQ(strlen(kCPUVendor), sizeof(cpu_vendor_registers));
  memcpy(cpu_vendor_registers, kCPUVendor, sizeof(cpu_vendor_registers));

  system_info_writer->SetCPUArchitecture(kCPUArchitecture);
  system_info_writer->SetCPULevelAndRevision(kCPULevel, kCPURevision);
  system_info_writer->SetCPUCount(kCPUCount);
  system_info_writer->SetOS(kOS);
  system_info_writer->SetOSType(kMinidumpOSTypeWorkstation);
  system_info_writer->SetOSVersion(
      kOSVersionMajor, kOSVersionMinor, kOSVersionBuild);
  system_info_writer->SetCSDVersion(kCSDVersion);
  system_info_writer->SetSuiteMask(kSuiteMask);
  system_info_writer->SetCPUX86VendorString(kCPUVendor);
  system_info_writer->SetCPUX86VersionAndFeatures(kCPUVersion, kCPUFeatures);
  system_info_writer->SetCPUX86AMDExtendedFeatures(kAMDFeatures);

  ASSERT_TRUE(minidump_file_writer.AddStream(std::move(system_info_writer)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  const MINIDUMP_SYSTEM_INFO* system_info = nullptr;
  const MINIDUMP_STRING* csd_version = nullptr;

  ASSERT_NO_FATAL_FAILURE(GetSystemInfoStream(
      string_file.string(), strlen(kCSDVersion), &system_info, &csd_version));

  EXPECT_EQ(system_info->ProcessorArchitecture, kCPUArchitecture);
  EXPECT_EQ(system_info->ProcessorLevel, kCPULevel);
  EXPECT_EQ(system_info->ProcessorRevision, kCPURevision);
  EXPECT_EQ(system_info->NumberOfProcessors, kCPUCount);
  EXPECT_EQ(system_info->ProductType, kOSType);
  EXPECT_EQ(system_info->MajorVersion, kOSVersionMajor);
  EXPECT_EQ(system_info->MinorVersion, kOSVersionMinor);
  EXPECT_EQ(system_info->BuildNumber, kOSVersionBuild);
  EXPECT_EQ(system_info->PlatformId, kOS);
  EXPECT_EQ(system_info->SuiteMask, kSuiteMask);
  EXPECT_EQ(system_info->Cpu.X86CpuInfo.VendorId[0], cpu_vendor_registers[0]);
  EXPECT_EQ(system_info->Cpu.X86CpuInfo.VendorId[1], cpu_vendor_registers[1]);
  EXPECT_EQ(system_info->Cpu.X86CpuInfo.VendorId[2], cpu_vendor_registers[2]);
  EXPECT_EQ(system_info->Cpu.X86CpuInfo.VersionInformation, kCPUVersion);
  EXPECT_EQ(system_info->Cpu.X86CpuInfo.FeatureInformation, kCPUFeatures);
  EXPECT_EQ(system_info->Cpu.X86CpuInfo.AMDExtendedCpuFeatures, kAMDFeatures);

  for (size_t index = 0; index < strlen(kCSDVersion); ++index) {
    EXPECT_EQ(csd_version->Buffer[index], kCSDVersion[index]) << index;
  }
}

TEST(MinidumpSystemInfoWriter, AMD64_Mac) {
  MinidumpFileWriter minidump_file_writer;
  auto system_info_writer = std::make_unique<MinidumpSystemInfoWriter>();

  constexpr MinidumpCPUArchitecture kCPUArchitecture =
      kMinidumpCPUArchitectureAMD64;
  constexpr uint16_t kCPULevel = 0x0006;
  constexpr uint16_t kCPURevision = 0x3a09;
  constexpr uint8_t kCPUCount = 8;
  constexpr MinidumpOS kOS = kMinidumpOSMacOSX;
  constexpr MinidumpOSType kOSType = kMinidumpOSTypeWorkstation;
  constexpr uint32_t kOSVersionMajor = 10;
  constexpr uint32_t kOSVersionMinor = 9;
  constexpr uint32_t kOSVersionBuild = 4;
  static constexpr char kCSDVersion[] = "13E28";
  static constexpr uint64_t kCPUFeatures[2] = {0x10427f4c, 0x00000000};

  system_info_writer->SetCPUArchitecture(kCPUArchitecture);
  system_info_writer->SetCPULevelAndRevision(kCPULevel, kCPURevision);
  system_info_writer->SetCPUCount(kCPUCount);
  system_info_writer->SetOS(kOS);
  system_info_writer->SetOSType(kMinidumpOSTypeWorkstation);
  system_info_writer->SetOSVersion(
      kOSVersionMajor, kOSVersionMinor, kOSVersionBuild);
  system_info_writer->SetCSDVersion(kCSDVersion);
  system_info_writer->SetCPUOtherFeatures(kCPUFeatures[0], kCPUFeatures[1]);

  ASSERT_TRUE(minidump_file_writer.AddStream(std::move(system_info_writer)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  const MINIDUMP_SYSTEM_INFO* system_info = nullptr;
  const MINIDUMP_STRING* csd_version;

  ASSERT_NO_FATAL_FAILURE(GetSystemInfoStream(
      string_file.string(), strlen(kCSDVersion), &system_info, &csd_version));

  EXPECT_EQ(system_info->ProcessorArchitecture, kCPUArchitecture);
  EXPECT_EQ(system_info->ProcessorLevel, kCPULevel);
  EXPECT_EQ(system_info->ProcessorRevision, kCPURevision);
  EXPECT_EQ(system_info->NumberOfProcessors, kCPUCount);
  EXPECT_EQ(system_info->ProductType, kOSType);
  EXPECT_EQ(system_info->MajorVersion, kOSVersionMajor);
  EXPECT_EQ(system_info->MinorVersion, kOSVersionMinor);
  EXPECT_EQ(system_info->BuildNumber, kOSVersionBuild);
  EXPECT_EQ(system_info->PlatformId, kOS);
  EXPECT_EQ(system_info->SuiteMask, 0u);
  EXPECT_EQ(system_info->Cpu.OtherCpuInfo.ProcessorFeatures[0],
            kCPUFeatures[0]);
  EXPECT_EQ(system_info->Cpu.OtherCpuInfo.ProcessorFeatures[1],
            kCPUFeatures[1]);
}

TEST(MinidumpSystemInfoWriter, X86_CPUVendorFromRegisters) {
  // MinidumpSystemInfoWriter.X86_Win already tested SetCPUX86VendorString().
  // This test exercises SetCPUX86Vendor() to set the vendor from register
  // values.
  MinidumpFileWriter minidump_file_writer;
  auto system_info_writer = std::make_unique<MinidumpSystemInfoWriter>();

  constexpr MinidumpCPUArchitecture kCPUArchitecture =
      kMinidumpCPUArchitectureX86;
  static constexpr uint32_t kCPUVendor[] = {'uneG', 'Ieni', 'letn'};

  system_info_writer->SetCPUArchitecture(kCPUArchitecture);
  system_info_writer->SetCPUX86Vendor(
      kCPUVendor[0], kCPUVendor[1], kCPUVendor[2]);
  system_info_writer->SetCSDVersion(std::string());

  ASSERT_TRUE(minidump_file_writer.AddStream(std::move(system_info_writer)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  const MINIDUMP_SYSTEM_INFO* system_info = nullptr;
  const MINIDUMP_STRING* csd_version;

  ASSERT_NO_FATAL_FAILURE(
      GetSystemInfoStream(string_file.string(), 0, &system_info, &csd_version));

  EXPECT_EQ(system_info->ProcessorArchitecture, kCPUArchitecture);
  EXPECT_EQ(system_info->ProcessorLevel, 0u);
  EXPECT_EQ(system_info->Cpu.X86CpuInfo.VendorId[0], kCPUVendor[0]);
  EXPECT_EQ(system_info->Cpu.X86CpuInfo.VendorId[1], kCPUVendor[1]);
  EXPECT_EQ(system_info->Cpu.X86CpuInfo.VendorId[2], kCPUVendor[2]);
  EXPECT_EQ(system_info->Cpu.X86CpuInfo.VersionInformation, 0u);
}

TEST(MinidumpSystemInfoWriter, InitializeFromSnapshot_X86) {
  MINIDUMP_SYSTEM_INFO expect_system_info = {};

  constexpr uint16_t kCPUFamily = 6;
  constexpr uint8_t kCPUModel = 70;
  constexpr uint8_t kCPUStepping = 1;

  const uint8_t kCPUBasicFamily =
      static_cast<uint8_t>(std::min(kCPUFamily, static_cast<uint16_t>(15)));
  const uint8_t kCPUExtendedFamily = kCPUFamily - kCPUBasicFamily;

  // These checks ensure that even if the constants above change, they represent
  // something that can legitimately be encoded in the form used by cpuid 1 eax.
  EXPECT_LE(kCPUFamily, 270);
  EXPECT_LE(kCPUStepping, 15);
  EXPECT_TRUE(kCPUBasicFamily == 6 || kCPUBasicFamily == 15 || kCPUModel <= 15);

  constexpr uint8_t kCPUBasicModel = kCPUModel & 0xf;
  constexpr uint8_t kCPUExtendedModel = kCPUModel >> 4;
  const uint32_t kCPUSignature =
      (kCPUExtendedFamily << 20) | (kCPUExtendedModel << 16) |
      (kCPUBasicFamily << 8) | (kCPUBasicModel << 4) | kCPUStepping;
  constexpr uint64_t kCPUX86Features = 0x7ffafbffbfebfbff;
  expect_system_info.ProcessorArchitecture = kMinidumpCPUArchitectureX86;
  expect_system_info.ProcessorLevel = kCPUFamily;
  expect_system_info.ProcessorRevision = (kCPUModel << 8) | kCPUStepping;
  expect_system_info.NumberOfProcessors = 8;
  expect_system_info.ProductType = kMinidumpOSTypeServer;
  expect_system_info.MajorVersion = 10;
  expect_system_info.MinorVersion = 9;
  expect_system_info.BuildNumber = 5;
  expect_system_info.PlatformId = kMinidumpOSMacOSX;
  expect_system_info.SuiteMask = 0;
  expect_system_info.Cpu.X86CpuInfo.VendorId[0] = 'uneG';
  expect_system_info.Cpu.X86CpuInfo.VendorId[1] = 'Ieni';
  expect_system_info.Cpu.X86CpuInfo.VendorId[2] = 'letn';
  expect_system_info.Cpu.X86CpuInfo.VersionInformation = kCPUSignature;
  expect_system_info.Cpu.X86CpuInfo.FeatureInformation =
      kCPUX86Features & 0xffffffff;
  static constexpr char kCPUVendor[] = "GenuineIntel";
  static constexpr char kOSVersionBuild[] = "13F34";

  TestSystemSnapshot system_snapshot;
  system_snapshot.SetCPUArchitecture(kCPUArchitectureX86);
  system_snapshot.SetCPURevision(
      (kCPUFamily << 16) | (kCPUModel << 8) | kCPUStepping);
  system_snapshot.SetCPUCount(expect_system_info.NumberOfProcessors);
  system_snapshot.SetCPUVendor(kCPUVendor);
  system_snapshot.SetCPUX86Signature(kCPUSignature);
  system_snapshot.SetCPUX86Features(kCPUX86Features);
  system_snapshot.SetOperatingSystem(SystemSnapshot::kOperatingSystemMacOSX);
  system_snapshot.SetOSServer(true);
  system_snapshot.SetOSVersion(expect_system_info.MajorVersion,
                               expect_system_info.MinorVersion,
                               expect_system_info.BuildNumber,
                               kOSVersionBuild);

  auto system_info_writer = std::make_unique<MinidumpSystemInfoWriter>();
  system_info_writer->InitializeFromSnapshot(&system_snapshot);

  MinidumpFileWriter minidump_file_writer;
  ASSERT_TRUE(minidump_file_writer.AddStream(std::move(system_info_writer)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  const MINIDUMP_SYSTEM_INFO* system_info = nullptr;
  const MINIDUMP_STRING* csd_version = nullptr;
  ASSERT_NO_FATAL_FAILURE(GetSystemInfoStream(string_file.string(),
                                              strlen(kOSVersionBuild),
                                              &system_info,
                                              &csd_version));

  EXPECT_EQ(system_info->ProcessorArchitecture,
            expect_system_info.ProcessorArchitecture);
  EXPECT_EQ(system_info->ProcessorLevel, expect_system_info.ProcessorLevel);
  EXPECT_EQ(system_info->ProcessorRevision,
            expect_system_info.ProcessorRevision);
  EXPECT_EQ(system_info->NumberOfProcessors,
            expect_system_info.NumberOfProcessors);
  EXPECT_EQ(system_info->ProductType, expect_system_info.ProductType);
  EXPECT_EQ(system_info->MajorVersion, expect_system_info.MajorVersion);
  EXPECT_EQ(system_info->MinorVersion, expect_system_info.MinorVersion);
  EXPECT_EQ(system_info->BuildNumber, expect_system_info.BuildNumber);
  EXPECT_EQ(system_info->PlatformId, expect_system_info.PlatformId);
  EXPECT_EQ(system_info->SuiteMask, expect_system_info.SuiteMask);
  EXPECT_EQ(system_info->Cpu.X86CpuInfo.VendorId[0],
            expect_system_info.Cpu.X86CpuInfo.VendorId[0]);
  EXPECT_EQ(system_info->Cpu.X86CpuInfo.VendorId[1],
            expect_system_info.Cpu.X86CpuInfo.VendorId[1]);
  EXPECT_EQ(system_info->Cpu.X86CpuInfo.VendorId[2],
            expect_system_info.Cpu.X86CpuInfo.VendorId[2]);
  EXPECT_EQ(system_info->Cpu.X86CpuInfo.VersionInformation,
            expect_system_info.Cpu.X86CpuInfo.VersionInformation);
  EXPECT_EQ(system_info->Cpu.X86CpuInfo.FeatureInformation,
            expect_system_info.Cpu.X86CpuInfo.FeatureInformation);

  for (size_t index = 0; index < strlen(kOSVersionBuild); ++index) {
    EXPECT_EQ(csd_version->Buffer[index], kOSVersionBuild[index]) << index;
  }
}

TEST(MinidumpSystemInfoWriter, InitializeFromSnapshot_AMD64) {
  MINIDUMP_SYSTEM_INFO expect_system_info = {};

  constexpr uint8_t kCPUFamily = 6;
  constexpr uint8_t kCPUModel = 70;
  constexpr uint8_t kCPUStepping = 1;
  expect_system_info.ProcessorArchitecture = kMinidumpCPUArchitectureAMD64;
  expect_system_info.ProcessorLevel = kCPUFamily;
  expect_system_info.ProcessorRevision = (kCPUModel << 8) | kCPUStepping;
  expect_system_info.NumberOfProcessors = 8;
  expect_system_info.ProductType = kMinidumpOSTypeServer;
  expect_system_info.MajorVersion = 10;
  expect_system_info.MinorVersion = 9;
  expect_system_info.BuildNumber = 5;
  expect_system_info.PlatformId = kMinidumpOSMacOSX;
  expect_system_info.SuiteMask = 0;
  expect_system_info.Cpu.OtherCpuInfo.ProcessorFeatures[0] =
      (1 << PF_COMPARE_EXCHANGE_DOUBLE) |
      (1 << PF_MMX_INSTRUCTIONS_AVAILABLE) |
      (1 << PF_XMMI_INSTRUCTIONS_AVAILABLE) |
      (1 << PF_RDTSC_INSTRUCTION_AVAILABLE) |
      (1 << PF_PAE_ENABLED) |
      (1 << PF_XMMI64_INSTRUCTIONS_AVAILABLE) |
      (1 << PF_SSE_DAZ_MODE_AVAILABLE) |
      (1 << PF_NX_ENABLED) |
      (1 << PF_SSE3_INSTRUCTIONS_AVAILABLE) |
      (1 << PF_COMPARE_EXCHANGE128) |
      (1 << PF_XSAVE_ENABLED) |
      (1 << PF_RDWRFSGSBASE_AVAILABLE) |
      (1 << PF_RDRAND_INSTRUCTION_AVAILABLE) |
      (UINT64_C(1) << PF_RDTSCP_INSTRUCTION_AVAILABLE);
  expect_system_info.Cpu.OtherCpuInfo.ProcessorFeatures[1] = 0;
  static constexpr char kOSVersionBuild[] = "13F34";

  TestSystemSnapshot system_snapshot;
  system_snapshot.SetCPUArchitecture(kCPUArchitectureX86_64);
  system_snapshot.SetCPURevision(
      (kCPUFamily << 16) | (kCPUModel << 8) | kCPUStepping);
  system_snapshot.SetCPUCount(expect_system_info.NumberOfProcessors);
  system_snapshot.SetCPUX86Features(0x7ffafbffbfebfbff);
  system_snapshot.SetCPUX86ExtendedFeatures(0x000000212c100900);
  system_snapshot.SetCPUX86Leaf7Features(0x00002fbb);
  system_snapshot.SetCPUX86SupportsDAZ(true);
  system_snapshot.SetOperatingSystem(SystemSnapshot::kOperatingSystemMacOSX);
  system_snapshot.SetOSServer(true);
  system_snapshot.SetOSVersion(expect_system_info.MajorVersion,
                               expect_system_info.MinorVersion,
                               expect_system_info.BuildNumber,
                               kOSVersionBuild);
  system_snapshot.SetNXEnabled(true);

  auto system_info_writer = std::make_unique<MinidumpSystemInfoWriter>();
  system_info_writer->InitializeFromSnapshot(&system_snapshot);

  MinidumpFileWriter minidump_file_writer;
  ASSERT_TRUE(minidump_file_writer.AddStream(std::move(system_info_writer)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  const MINIDUMP_SYSTEM_INFO* system_info = nullptr;
  const MINIDUMP_STRING* csd_version = nullptr;
  ASSERT_NO_FATAL_FAILURE(GetSystemInfoStream(string_file.string(),
                                              strlen(kOSVersionBuild),
                                              &system_info,
                                              &csd_version));

  EXPECT_EQ(system_info->ProcessorArchitecture,
            expect_system_info.ProcessorArchitecture);
  EXPECT_EQ(system_info->ProcessorLevel, expect_system_info.ProcessorLevel);
  EXPECT_EQ(system_info->ProcessorRevision,
            expect_system_info.ProcessorRevision);
  EXPECT_EQ(system_info->NumberOfProcessors,
            expect_system_info.NumberOfProcessors);
  EXPECT_EQ(system_info->ProductType, expect_system_info.ProductType);
  EXPECT_EQ(system_info->MajorVersion, expect_system_info.MajorVersion);
  EXPECT_EQ(system_info->MinorVersion, expect_system_info.MinorVersion);
  EXPECT_EQ(system_info->BuildNumber, expect_system_info.BuildNumber);
  EXPECT_EQ(system_info->PlatformId, expect_system_info.PlatformId);
  EXPECT_EQ(system_info->SuiteMask, expect_system_info.SuiteMask);
  EXPECT_EQ(system_info->Cpu.OtherCpuInfo.ProcessorFeatures[0],
            expect_system_info.Cpu.OtherCpuInfo.ProcessorFeatures[0]);
  EXPECT_EQ(system_info->Cpu.OtherCpuInfo.ProcessorFeatures[1],
            expect_system_info.Cpu.OtherCpuInfo.ProcessorFeatures[1]);

  for (size_t index = 0; index < strlen(kOSVersionBuild); ++index) {
    EXPECT_EQ(csd_version->Buffer[index], kOSVersionBuild[index]) << index;
  }
}

TEST(MinidumpSystemInfoWriterDeathTest, NoCSDVersion) {
  MinidumpFileWriter minidump_file_writer;
  auto system_info_writer = std::make_unique<MinidumpSystemInfoWriter>();
  ASSERT_TRUE(minidump_file_writer.AddStream(std::move(system_info_writer)));

  StringFile string_file;
  ASSERT_DEATH_CHECK(minidump_file_writer.WriteEverything(&string_file),
                     "csd_version_");
}

}  // namespace
}  // namespace test
}  // namespace crashpad
