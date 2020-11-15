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

#include "minidump/minidump_module_writer.h"

#include <stddef.h>
#include <string.h>

#include <utility>

#include "base/format_macros.h"
#include "base/strings/stringprintf.h"
#include "base/strings/utf_string_conversions.h"
#include "gtest/gtest.h"
#include "minidump/minidump_file_writer.h"
#include "minidump/test/minidump_file_writer_test_util.h"
#include "minidump/test/minidump_string_writer_test_util.h"
#include "minidump/test/minidump_writable_test_util.h"
#include "snapshot/test/test_module_snapshot.h"
#include "test/gtest_death.h"
#include "util/file/string_file.h"
#include "util/misc/implicit_cast.h"
#include "util/misc/uuid.h"

namespace crashpad {
namespace test {
namespace {

void GetModuleListStream(const std::string& file_contents,
                         const MINIDUMP_MODULE_LIST** module_list) {
  constexpr size_t kDirectoryOffset = sizeof(MINIDUMP_HEADER);
  constexpr size_t kModuleListStreamOffset =
      kDirectoryOffset + sizeof(MINIDUMP_DIRECTORY);
  constexpr size_t kModulesOffset =
      kModuleListStreamOffset + sizeof(MINIDUMP_MODULE_LIST);

  ASSERT_GE(file_contents.size(), kModulesOffset);

  const MINIDUMP_DIRECTORY* directory;
  const MINIDUMP_HEADER* header =
      MinidumpHeaderAtStart(file_contents, &directory);
  ASSERT_NO_FATAL_FAILURE(VerifyMinidumpHeader(header, 1, 0));
  ASSERT_TRUE(directory);

  ASSERT_EQ(directory[0].StreamType, kMinidumpStreamTypeModuleList);
  EXPECT_EQ(directory[0].Location.Rva, kModuleListStreamOffset);

  *module_list = MinidumpWritableAtLocationDescriptor<MINIDUMP_MODULE_LIST>(
      file_contents, directory[0].Location);
  ASSERT_TRUE(module_list);
}

TEST(MinidumpModuleWriter, EmptyModuleList) {
  MinidumpFileWriter minidump_file_writer;
  auto module_list_writer = std::make_unique<MinidumpModuleListWriter>();

  ASSERT_TRUE(minidump_file_writer.AddStream(std::move(module_list_writer)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  ASSERT_EQ(string_file.string().size(),
            sizeof(MINIDUMP_HEADER) + sizeof(MINIDUMP_DIRECTORY) +
                sizeof(MINIDUMP_MODULE_LIST));

  const MINIDUMP_MODULE_LIST* module_list = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      GetModuleListStream(string_file.string(), &module_list));

  EXPECT_EQ(module_list->NumberOfModules, 0u);
}

// If |expected_pdb_name| is not nullptr, |codeview_record| is used to locate a
// CodeView record in |file_contents|, and its fields are compared against the
// |expected_pdb_*| values. If |expected_pdb_uuid| is supplied, the CodeView
// record must be a PDB 7.0 link, otherwise, it must be a PDB 2.0 link. If
// |expected_pdb_name| is nullptr, |codeview_record| must not point to anything.
void ExpectCodeViewRecord(const MINIDUMP_LOCATION_DESCRIPTOR* codeview_record,
                          const std::string& file_contents,
                          const char* expected_pdb_name,
                          const UUID* expected_pdb_uuid,
                          time_t expected_pdb_timestamp,
                          uint32_t expected_pdb_age) {
  if (expected_pdb_name) {
    EXPECT_NE(codeview_record->Rva, 0u);

    std::string observed_pdb_name;
    if (expected_pdb_uuid) {
      // The CodeView record should be a PDB 7.0 link.
      const CodeViewRecordPDB70* codeview_pdb70_record =
          MinidumpWritableAtLocationDescriptor<CodeViewRecordPDB70>(
              file_contents, *codeview_record);
      ASSERT_TRUE(codeview_pdb70_record);
      EXPECT_EQ(memcmp(expected_pdb_uuid,
                       &codeview_pdb70_record->uuid,
                       sizeof(codeview_pdb70_record->uuid)),
                0);
      EXPECT_EQ(codeview_pdb70_record->age, expected_pdb_age);

      observed_pdb_name.assign(
          reinterpret_cast<const char*>(&codeview_pdb70_record->pdb_name[0]),
          codeview_record->DataSize - offsetof(CodeViewRecordPDB70, pdb_name));
    } else {
      // The CodeView record should be a PDB 2.0 link.
      const CodeViewRecordPDB20* codeview_pdb20_record =
          MinidumpWritableAtLocationDescriptor<CodeViewRecordPDB20>(
              file_contents, *codeview_record);
      ASSERT_TRUE(codeview_pdb20_record);
      EXPECT_EQ(codeview_pdb20_record->timestamp,
                static_cast<uint32_t>(expected_pdb_timestamp));
      EXPECT_EQ(codeview_pdb20_record->age, expected_pdb_age);

      observed_pdb_name.assign(
          reinterpret_cast<const char*>(&codeview_pdb20_record->pdb_name[0]),
          codeview_record->DataSize - offsetof(CodeViewRecordPDB20, pdb_name));
    }

    // Check for, and then remove, the NUL terminator.
    EXPECT_EQ(observed_pdb_name[observed_pdb_name.size() - 1], '\0');
    observed_pdb_name.resize(observed_pdb_name.size() - 1);

    EXPECT_EQ(observed_pdb_name, expected_pdb_name);
  } else {
    // There should be no CodeView record.
    EXPECT_EQ(codeview_record->DataSize, 0u);
    EXPECT_EQ(codeview_record->Rva, 0u);
  }
}

// If |expected_debug_name| is not nullptr, |misc_record| is used to locate a
// miscellanous debugging record in |file_contents|, and its fields are compared
// against the the |expected_debug_*| values. If |expected_debug_name| is
// nullptr, |misc_record| must not point to anything.
void ExpectMiscellaneousDebugRecord(
    const MINIDUMP_LOCATION_DESCRIPTOR* misc_record,
    const std::string& file_contents,
    const char* expected_debug_name,
    uint32_t expected_debug_type,
    bool expected_debug_utf16) {
  if (expected_debug_name) {
    EXPECT_NE(misc_record->Rva, 0u);
    const IMAGE_DEBUG_MISC* misc_debug_record =
        MinidumpWritableAtLocationDescriptor<IMAGE_DEBUG_MISC>(file_contents,
                                                               *misc_record);
    ASSERT_TRUE(misc_debug_record);
    EXPECT_EQ(misc_debug_record->DataType, expected_debug_type);
    EXPECT_EQ(misc_debug_record->Unicode != 0, expected_debug_utf16);
    EXPECT_EQ(misc_debug_record->Reserved[0], 0u);
    EXPECT_EQ(misc_debug_record->Reserved[1], 0u);
    EXPECT_EQ(misc_debug_record->Reserved[2], 0u);

    // Check for the NUL terminator.
    size_t bytes_available =
        misc_debug_record->Length - offsetof(IMAGE_DEBUG_MISC, Data);
    EXPECT_EQ(misc_debug_record->Data[bytes_available - 1], '\0');
    std::string observed_data(
        reinterpret_cast<const char*>(misc_debug_record->Data));

    size_t bytes_used;
    if (misc_debug_record->Unicode) {
      base::string16 observed_data_utf16(
          reinterpret_cast<const base::char16*>(misc_debug_record->Data));
      bytes_used = (observed_data_utf16.size() + 1) * sizeof(base::char16);
      observed_data = base::UTF16ToUTF8(observed_data_utf16);
    } else {
      observed_data = reinterpret_cast<const char*>(misc_debug_record->Data);
      bytes_used = (observed_data.size() + 1) * sizeof(char);
    }
    EXPECT_LE(bytes_used, bytes_available);

    // Make sure that any padding bytes after the first NUL are also NUL.
    for (size_t index = bytes_used; index < bytes_available; ++index) {
      EXPECT_EQ(misc_debug_record->Data[index], '\0');
    }

    EXPECT_EQ(observed_data, expected_debug_name);
  } else {
    // There should be no miscellaneous debugging record.
    EXPECT_EQ(misc_record->DataSize, 0u);
    EXPECT_EQ(misc_record->Rva, 0u);
  }
}

// ExpectModule() verifies that |expected| matches |observed|. Fields that are
// supposed to contain constant magic numbers are verified against the expected
// constants instead of |expected|. Reserved fields are verified to be 0. RVA
// and MINIDUMP_LOCATION_DESCRIPTOR fields are not verified against |expected|.
// Instead, |ModuleNameRva| is used to locate the module name, which is compared
// against |expected_module_name|. ExpectCodeViewRecord() and
// ExpectMiscellaneousDebugRecord() are used to verify the |CvRecord| and
// |MiscRecord| fields against |expected_pdb_*| and |expected_debug_*|
// parameters, respectively.
void ExpectModule(const MINIDUMP_MODULE* expected,
                  const MINIDUMP_MODULE* observed,
                  const std::string& file_contents,
                  const std::string& expected_module_name,
                  const char* expected_pdb_name,
                  const UUID* expected_pdb_uuid,
                  time_t expected_pdb_timestamp,
                  uint32_t expected_pdb_age,
                  const char* expected_debug_name,
                  uint32_t expected_debug_type,
                  bool expected_debug_utf16) {
  EXPECT_EQ(observed->BaseOfImage, expected->BaseOfImage);
  EXPECT_EQ(observed->SizeOfImage, expected->SizeOfImage);
  EXPECT_EQ(observed->CheckSum, expected->CheckSum);
  EXPECT_EQ(observed->TimeDateStamp, expected->TimeDateStamp);
  EXPECT_EQ(observed->VersionInfo.dwSignature,
            implicit_cast<uint32_t>(VS_FFI_SIGNATURE));
  EXPECT_EQ(observed->VersionInfo.dwStrucVersion,
            implicit_cast<uint32_t>(VS_FFI_STRUCVERSION));
  EXPECT_EQ(observed->VersionInfo.dwFileVersionMS,
            expected->VersionInfo.dwFileVersionMS);
  EXPECT_EQ(observed->VersionInfo.dwFileVersionLS,
            expected->VersionInfo.dwFileVersionLS);
  EXPECT_EQ(observed->VersionInfo.dwProductVersionMS,
            expected->VersionInfo.dwProductVersionMS);
  EXPECT_EQ(observed->VersionInfo.dwProductVersionLS,
            expected->VersionInfo.dwProductVersionLS);
  EXPECT_EQ(observed->VersionInfo.dwFileFlagsMask,
            expected->VersionInfo.dwFileFlagsMask);
  EXPECT_EQ(observed->VersionInfo.dwFileFlags,
            expected->VersionInfo.dwFileFlags);
  EXPECT_EQ(observed->VersionInfo.dwFileOS, expected->VersionInfo.dwFileOS);
  EXPECT_EQ(observed->VersionInfo.dwFileType, expected->VersionInfo.dwFileType);
  EXPECT_EQ(observed->VersionInfo.dwFileSubtype,
            expected->VersionInfo.dwFileSubtype);
  EXPECT_EQ(observed->VersionInfo.dwFileDateMS,
            expected->VersionInfo.dwFileDateMS);
  EXPECT_EQ(observed->VersionInfo.dwFileDateLS,
            expected->VersionInfo.dwFileDateLS);
  EXPECT_EQ(observed->Reserved0, 0u);
  EXPECT_EQ(observed->Reserved1, 0u);

  EXPECT_NE(observed->ModuleNameRva, 0u);
  base::string16 observed_module_name_utf16 =
      MinidumpStringAtRVAAsString(file_contents, observed->ModuleNameRva);
  base::string16 expected_module_name_utf16 =
      base::UTF8ToUTF16(expected_module_name);
  EXPECT_EQ(observed_module_name_utf16, expected_module_name_utf16);

  ASSERT_NO_FATAL_FAILURE(ExpectCodeViewRecord(&observed->CvRecord,
                                               file_contents,
                                               expected_pdb_name,
                                               expected_pdb_uuid,
                                               expected_pdb_timestamp,
                                               expected_pdb_age));

  ASSERT_NO_FATAL_FAILURE(ExpectMiscellaneousDebugRecord(&observed->MiscRecord,
                                                         file_contents,
                                                         expected_debug_name,
                                                         expected_debug_type,
                                                         expected_debug_utf16));
}

TEST(MinidumpModuleWriter, EmptyModule) {
  MinidumpFileWriter minidump_file_writer;
  auto module_list_writer = std::make_unique<MinidumpModuleListWriter>();

  static constexpr char kModuleName[] = "test_executable";

  auto module_writer = std::make_unique<MinidumpModuleWriter>();
  module_writer->SetName(kModuleName);

  module_list_writer->AddModule(std::move(module_writer));
  ASSERT_TRUE(minidump_file_writer.AddStream(std::move(module_list_writer)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  ASSERT_GT(string_file.string().size(),
            sizeof(MINIDUMP_HEADER) + sizeof(MINIDUMP_DIRECTORY) +
                sizeof(MINIDUMP_MODULE_LIST) + 1 * sizeof(MINIDUMP_MODULE));

  const MINIDUMP_MODULE_LIST* module_list = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      GetModuleListStream(string_file.string(), &module_list));

  EXPECT_EQ(module_list->NumberOfModules, 1u);

  MINIDUMP_MODULE expected = {};
  ASSERT_NO_FATAL_FAILURE(ExpectModule(&expected,
                                       &module_list->Modules[0],
                                       string_file.string(),
                                       kModuleName,
                                       nullptr,
                                       nullptr,
                                       0,
                                       0,
                                       nullptr,
                                       0,
                                       false));
}

TEST(MinidumpModuleWriter, OneModule) {
  MinidumpFileWriter minidump_file_writer;
  auto module_list_writer = std::make_unique<MinidumpModuleListWriter>();

  static constexpr char kModuleName[] = "statically_linked";
  constexpr uint64_t kModuleBase = 0x10da69000;
  constexpr uint32_t kModuleSize = 0x1000;
  constexpr uint32_t kChecksum = 0x76543210;
  constexpr time_t kTimestamp = 0x386d4380;
  constexpr uint32_t kFileVersionMS = 0x00010002;
  constexpr uint32_t kFileVersionLS = 0x00030004;
  constexpr uint32_t kProductVersionMS = 0x00050006;
  constexpr uint32_t kProductVersionLS = 0x00070008;
  constexpr uint32_t kFileFlagsMask = VS_FF_DEBUG | VS_FF_PRERELEASE |
                                      VS_FF_PATCHED | VS_FF_PRIVATEBUILD |
                                      VS_FF_INFOINFERRED | VS_FF_SPECIALBUILD;
  constexpr uint32_t kFileFlags = VS_FF_PRIVATEBUILD | VS_FF_SPECIALBUILD;
  constexpr uint32_t kFileOS = VOS_DOS;
  constexpr uint32_t kFileType = VFT_DRV;
  constexpr uint32_t kFileSubtype = VFT2_DRV_KEYBOARD;
  static constexpr char kPDBName[] = "statical.pdb";
  static constexpr uint8_t kPDBUUIDBytes[16] =
      {0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10,
       0x08, 0x19, 0x2a, 0x3b, 0x4c, 0x5d, 0x6e, 0x7f};
  UUID pdb_uuid;
  pdb_uuid.InitializeFromBytes(kPDBUUIDBytes);
  constexpr uint32_t kPDBAge = 1;
  constexpr uint32_t kDebugType = IMAGE_DEBUG_MISC_EXENAME;
  static constexpr char kDebugName[] = "statical.dbg";
  constexpr bool kDebugUTF16 = false;

  auto module_writer = std::make_unique<MinidumpModuleWriter>();
  module_writer->SetName(kModuleName);
  module_writer->SetImageBaseAddress(kModuleBase);
  module_writer->SetImageSize(kModuleSize);
  module_writer->SetChecksum(kChecksum);
  module_writer->SetTimestamp(kTimestamp);
  module_writer->SetFileVersion(kFileVersionMS >> 16,
                                kFileVersionMS & 0xffff,
                                kFileVersionLS >> 16,
                                kFileVersionLS & 0xffff);
  module_writer->SetProductVersion(kProductVersionMS >> 16,
                                   kProductVersionMS & 0xffff,
                                   kProductVersionLS >> 16,
                                   kProductVersionLS & 0xffff);
  module_writer->SetFileFlagsAndMask(kFileFlags, kFileFlagsMask);
  module_writer->SetFileOS(kFileOS);
  module_writer->SetFileTypeAndSubtype(kFileType, kFileSubtype);

  auto codeview_pdb70_writer =
      std::make_unique<MinidumpModuleCodeViewRecordPDB70Writer>();
  codeview_pdb70_writer->SetPDBName(kPDBName);
  codeview_pdb70_writer->SetUUIDAndAge(pdb_uuid, kPDBAge);
  module_writer->SetCodeViewRecord(std::move(codeview_pdb70_writer));

  auto misc_debug_writer =
      std::make_unique<MinidumpModuleMiscDebugRecordWriter>();
  misc_debug_writer->SetDataType(kDebugType);
  misc_debug_writer->SetData(kDebugName, kDebugUTF16);
  module_writer->SetMiscDebugRecord(std::move(misc_debug_writer));

  module_list_writer->AddModule(std::move(module_writer));
  ASSERT_TRUE(minidump_file_writer.AddStream(std::move(module_list_writer)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  ASSERT_GT(string_file.string().size(),
            sizeof(MINIDUMP_HEADER) + sizeof(MINIDUMP_DIRECTORY) +
                sizeof(MINIDUMP_MODULE_LIST) + 1 * sizeof(MINIDUMP_MODULE));

  const MINIDUMP_MODULE_LIST* module_list = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      GetModuleListStream(string_file.string(), &module_list));

  EXPECT_EQ(module_list->NumberOfModules, 1u);

  MINIDUMP_MODULE expected = {};
  expected.BaseOfImage = kModuleBase;
  expected.SizeOfImage = kModuleSize;
  expected.CheckSum = kChecksum;
  expected.TimeDateStamp = kTimestamp;
  expected.VersionInfo.dwFileVersionMS = kFileVersionMS;
  expected.VersionInfo.dwFileVersionLS = kFileVersionLS;
  expected.VersionInfo.dwProductVersionMS = kProductVersionMS;
  expected.VersionInfo.dwProductVersionLS = kProductVersionLS;
  expected.VersionInfo.dwFileFlagsMask = kFileFlagsMask;
  expected.VersionInfo.dwFileFlags = kFileFlags;
  expected.VersionInfo.dwFileOS = kFileOS;
  expected.VersionInfo.dwFileType = kFileType;
  expected.VersionInfo.dwFileSubtype = kFileSubtype;

  ASSERT_NO_FATAL_FAILURE(ExpectModule(&expected,
                                       &module_list->Modules[0],
                                       string_file.string(),
                                       kModuleName,
                                       kPDBName,
                                       &pdb_uuid,
                                       0,
                                       kPDBAge,
                                       kDebugName,
                                       kDebugType,
                                       kDebugUTF16));
}

TEST(MinidumpModuleWriter, OneModule_CodeViewUsesPDB20_MiscUsesUTF16) {
  // MinidumpModuleWriter.OneModule tested with a PDB 7.0 link as the CodeView
  // record and an IMAGE_DEBUG_MISC record in UTF-8. This test exercises the
  // alternatives, a PDB 2.0 link as the CodeView record and an IMAGE_DEBUG_MISC
  // record with UTF-16 data.
  MinidumpFileWriter minidump_file_writer;
  auto module_list_writer = std::make_unique<MinidumpModuleListWriter>();

  static constexpr char kModuleName[] = "dinosaur";
  static constexpr char kPDBName[] = "d1n05.pdb";
  constexpr time_t kPDBTimestamp = 0x386d4380;
  constexpr uint32_t kPDBAge = 1;
  constexpr uint32_t kDebugType = IMAGE_DEBUG_MISC_EXENAME;
  static constexpr char kDebugName[] = "d1n05.dbg";
  constexpr bool kDebugUTF16 = true;

  auto module_writer = std::make_unique<MinidumpModuleWriter>();
  module_writer->SetName(kModuleName);

  auto codeview_pdb20_writer =
      std::make_unique<MinidumpModuleCodeViewRecordPDB20Writer>();
  codeview_pdb20_writer->SetPDBName(kPDBName);
  codeview_pdb20_writer->SetTimestampAndAge(kPDBTimestamp, kPDBAge);
  module_writer->SetCodeViewRecord(std::move(codeview_pdb20_writer));

  auto misc_debug_writer =
      std::make_unique<MinidumpModuleMiscDebugRecordWriter>();
  misc_debug_writer->SetDataType(kDebugType);
  misc_debug_writer->SetData(kDebugName, kDebugUTF16);
  module_writer->SetMiscDebugRecord(std::move(misc_debug_writer));

  module_list_writer->AddModule(std::move(module_writer));
  ASSERT_TRUE(minidump_file_writer.AddStream(std::move(module_list_writer)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  ASSERT_GT(string_file.string().size(),
            sizeof(MINIDUMP_HEADER) + sizeof(MINIDUMP_DIRECTORY) +
                sizeof(MINIDUMP_MODULE_LIST) + 1 * sizeof(MINIDUMP_MODULE));

  const MINIDUMP_MODULE_LIST* module_list = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      GetModuleListStream(string_file.string(), &module_list));

  EXPECT_EQ(module_list->NumberOfModules, 1u);

  MINIDUMP_MODULE expected = {};

  ASSERT_NO_FATAL_FAILURE(ExpectModule(&expected,
                                       &module_list->Modules[0],
                                       string_file.string(),
                                       kModuleName,
                                       kPDBName,
                                       nullptr,
                                       kPDBTimestamp,
                                       kPDBAge,
                                       kDebugName,
                                       kDebugType,
                                       kDebugUTF16));
}

TEST(MinidumpModuleWriter, ThreeModules) {
  // As good exercise, this test uses three modules, one with a PDB 7.0 link as
  // its CodeView record, one with no CodeView record, and one with a PDB 2.0
  // link as its CodeView record.
  MinidumpFileWriter minidump_file_writer;
  auto module_list_writer = std::make_unique<MinidumpModuleListWriter>();

  static constexpr char kModuleName0[] = "main";
  constexpr uint64_t kModuleBase0 = 0x100101000;
  constexpr uint32_t kModuleSize0 = 0xf000;
  static constexpr char kPDBName0[] = "main";
  static constexpr uint8_t kPDBUUIDBytes0[16] =
      {0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00, 0x11,
       0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99};
  UUID pdb_uuid_0;
  pdb_uuid_0.InitializeFromBytes(kPDBUUIDBytes0);
  constexpr uint32_t kPDBAge0 = 0;

  static constexpr char kModuleName1[] = "ld.so";
  constexpr uint64_t kModuleBase1 = 0x200202000;
  constexpr uint32_t kModuleSize1 = 0x1e000;

  static constexpr char kModuleName2[] = "libc.so";
  constexpr uint64_t kModuleBase2 = 0x300303000;
  constexpr uint32_t kModuleSize2 = 0x2d000;
  static constexpr char kPDBName2[] = "libc.so";
  constexpr time_t kPDBTimestamp2 = 0x386d4380;
  constexpr uint32_t kPDBAge2 = 2;

  auto module_writer_0 = std::make_unique<MinidumpModuleWriter>();
  module_writer_0->SetName(kModuleName0);
  module_writer_0->SetImageBaseAddress(kModuleBase0);
  module_writer_0->SetImageSize(kModuleSize0);

  auto codeview_pdb70_writer_0 =
      std::make_unique<MinidumpModuleCodeViewRecordPDB70Writer>();
  codeview_pdb70_writer_0->SetPDBName(kPDBName0);
  codeview_pdb70_writer_0->SetUUIDAndAge(pdb_uuid_0, kPDBAge0);
  module_writer_0->SetCodeViewRecord(std::move(codeview_pdb70_writer_0));

  module_list_writer->AddModule(std::move(module_writer_0));

  auto module_writer_1 = std::make_unique<MinidumpModuleWriter>();
  module_writer_1->SetName(kModuleName1);
  module_writer_1->SetImageBaseAddress(kModuleBase1);
  module_writer_1->SetImageSize(kModuleSize1);

  module_list_writer->AddModule(std::move(module_writer_1));

  auto module_writer_2 = std::make_unique<MinidumpModuleWriter>();
  module_writer_2->SetName(kModuleName2);
  module_writer_2->SetImageBaseAddress(kModuleBase2);
  module_writer_2->SetImageSize(kModuleSize2);

  auto codeview_pdb70_writer_2 =
      std::make_unique<MinidumpModuleCodeViewRecordPDB20Writer>();
  codeview_pdb70_writer_2->SetPDBName(kPDBName2);
  codeview_pdb70_writer_2->SetTimestampAndAge(kPDBTimestamp2, kPDBAge2);
  module_writer_2->SetCodeViewRecord(std::move(codeview_pdb70_writer_2));

  module_list_writer->AddModule(std::move(module_writer_2));

  ASSERT_TRUE(minidump_file_writer.AddStream(std::move(module_list_writer)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  ASSERT_GT(string_file.string().size(),
            sizeof(MINIDUMP_HEADER) + sizeof(MINIDUMP_DIRECTORY) +
                sizeof(MINIDUMP_MODULE_LIST) + 1 * sizeof(MINIDUMP_MODULE));

  const MINIDUMP_MODULE_LIST* module_list = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      GetModuleListStream(string_file.string(), &module_list));

  EXPECT_EQ(module_list->NumberOfModules, 3u);

  MINIDUMP_MODULE expected = {};

  {
    SCOPED_TRACE("module 0");

    expected.BaseOfImage = kModuleBase0;
    expected.SizeOfImage = kModuleSize0;

    ASSERT_NO_FATAL_FAILURE(ExpectModule(&expected,
                                         &module_list->Modules[0],
                                         string_file.string(),
                                         kModuleName0,
                                         kPDBName0,
                                         &pdb_uuid_0,
                                         0,
                                         kPDBAge0,
                                         nullptr,
                                         0,
                                         false));
  }

  {
    SCOPED_TRACE("module 1");

    expected.BaseOfImage = kModuleBase1;
    expected.SizeOfImage = kModuleSize1;

    ASSERT_NO_FATAL_FAILURE(ExpectModule(&expected,
                                         &module_list->Modules[1],
                                         string_file.string(),
                                         kModuleName1,
                                         nullptr,
                                         nullptr,
                                         0,
                                         0,
                                         nullptr,
                                         0,
                                         false));
  }

  {
    SCOPED_TRACE("module 2");

    expected.BaseOfImage = kModuleBase2;
    expected.SizeOfImage = kModuleSize2;

    ASSERT_NO_FATAL_FAILURE(ExpectModule(&expected,
                                         &module_list->Modules[2],
                                         string_file.string(),
                                         kModuleName2,
                                         kPDBName2,
                                         nullptr,
                                         kPDBTimestamp2,
                                         kPDBAge2,
                                         nullptr,
                                         0,
                                         false));
  }
}

void InitializeTestModuleSnapshotFromMinidumpModule(
    TestModuleSnapshot* module_snapshot,
    const MINIDUMP_MODULE& minidump_module,
    const std::string& name,
    const std::string& pdb_name,
    const crashpad::UUID& uuid,
    uint32_t age) {
  module_snapshot->SetName(name);

  module_snapshot->SetAddressAndSize(minidump_module.BaseOfImage,
                                     minidump_module.SizeOfImage);
  module_snapshot->SetTimestamp(minidump_module.TimeDateStamp);
  module_snapshot->SetFileVersion(
      minidump_module.VersionInfo.dwFileVersionMS >> 16,
      minidump_module.VersionInfo.dwFileVersionMS & 0xffff,
      minidump_module.VersionInfo.dwFileVersionLS >> 16,
      minidump_module.VersionInfo.dwFileVersionLS & 0xffff);
  module_snapshot->SetSourceVersion(
      minidump_module.VersionInfo.dwProductVersionMS >> 16,
      minidump_module.VersionInfo.dwProductVersionMS & 0xffff,
      minidump_module.VersionInfo.dwProductVersionLS >> 16,
      minidump_module.VersionInfo.dwProductVersionLS & 0xffff);

  ModuleSnapshot::ModuleType module_type;
  switch (minidump_module.VersionInfo.dwFileType) {
    case VFT_APP:
      module_type = ModuleSnapshot::kModuleTypeExecutable;
      break;
    case VFT_DLL:
      module_type = ModuleSnapshot::kModuleTypeSharedLibrary;
      break;
    default:
      module_type = ModuleSnapshot::kModuleTypeUnknown;
      break;
  }
  module_snapshot->SetModuleType(module_type);

  module_snapshot->SetUUIDAndAge(uuid, age);
  module_snapshot->SetDebugFileName(pdb_name);
}

TEST(MinidumpModuleWriter, InitializeFromSnapshot) {
  MINIDUMP_MODULE expect_modules[3] = {};
  const char* module_paths[arraysize(expect_modules)] = {};
  const char* module_pdbs[arraysize(expect_modules)] = {};
  UUID uuids[arraysize(expect_modules)] = {};
  uint32_t ages[arraysize(expect_modules)] = {};

  expect_modules[0].BaseOfImage = 0x100101000;
  expect_modules[0].SizeOfImage = 0xf000;
  expect_modules[0].TimeDateStamp = 0x01234567;
  expect_modules[0].VersionInfo.dwFileVersionMS = 0x00010002;
  expect_modules[0].VersionInfo.dwFileVersionLS = 0x00030004;
  expect_modules[0].VersionInfo.dwProductVersionMS = 0x00050006;
  expect_modules[0].VersionInfo.dwProductVersionLS = 0x00070008;
  expect_modules[0].VersionInfo.dwFileType = VFT_APP;
  module_paths[0] = "/usr/bin/true";
  module_pdbs[0] = "true";
  static constexpr uint8_t kUUIDBytes0[16] =
      {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
       0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
  uuids[0].InitializeFromBytes(kUUIDBytes0);
  ages[0] = 10;

  expect_modules[1].BaseOfImage = 0x200202000;
  expect_modules[1].SizeOfImage = 0x1e1000;
  expect_modules[1].TimeDateStamp = 0x89abcdef;
  expect_modules[1].VersionInfo.dwFileVersionMS = 0x0009000a;
  expect_modules[1].VersionInfo.dwFileVersionLS = 0x000b000c;
  expect_modules[1].VersionInfo.dwProductVersionMS = 0x000d000e;
  expect_modules[1].VersionInfo.dwProductVersionLS = 0x000f0000;
  expect_modules[1].VersionInfo.dwFileType = VFT_DLL;
  module_paths[1] = "/usr/lib/libSystem.B.dylib";
  module_pdbs[1] = "libSystem.B.dylib.pdb";
  static constexpr uint8_t kUUIDBytes1[16] =
      {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
       0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
  uuids[1].InitializeFromBytes(kUUIDBytes1);
  ages[1] = 20;

  expect_modules[2].BaseOfImage = 0x300303000;
  expect_modules[2].SizeOfImage = 0x2d000;
  expect_modules[2].TimeDateStamp = 0x76543210;
  expect_modules[2].VersionInfo.dwFileVersionMS = 0x11112222;
  expect_modules[2].VersionInfo.dwFileVersionLS = 0x33334444;
  expect_modules[2].VersionInfo.dwProductVersionMS = 0x9999aaaa;
  expect_modules[2].VersionInfo.dwProductVersionLS = 0xbbbbcccc;
  expect_modules[2].VersionInfo.dwFileType = VFT_UNKNOWN;
  module_paths[2] = "/usr/lib/dyld";
  module_pdbs[2] = "/usr/lib/dyld.pdb";
  static constexpr uint8_t kUUIDBytes2[16] =
      {0xff, 0xfe, 0xfd, 0xfc, 0xfb, 0xfa, 0xf9, 0xf8,
       0xf7, 0xf6, 0xf5, 0xf4, 0xf3, 0xf2, 0xf1, 0xf0};
  uuids[2].InitializeFromBytes(kUUIDBytes2);
  ages[2] = 30;

  std::vector<std::unique_ptr<TestModuleSnapshot>> module_snapshots_owner;
  std::vector<const ModuleSnapshot*> module_snapshots;
  for (size_t index = 0; index < arraysize(expect_modules); ++index) {
    module_snapshots_owner.push_back(std::make_unique<TestModuleSnapshot>());
    TestModuleSnapshot* module_snapshot = module_snapshots_owner.back().get();
    InitializeTestModuleSnapshotFromMinidumpModule(module_snapshot,
                                                   expect_modules[index],
                                                   module_paths[index],
                                                   module_pdbs[index],
                                                   uuids[index],
                                                   ages[index]);
    module_snapshots.push_back(module_snapshot);
  }

  auto module_list_writer = std::make_unique<MinidumpModuleListWriter>();
  module_list_writer->InitializeFromSnapshot(module_snapshots);

  MinidumpFileWriter minidump_file_writer;
  ASSERT_TRUE(minidump_file_writer.AddStream(std::move(module_list_writer)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  const MINIDUMP_MODULE_LIST* module_list = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      GetModuleListStream(string_file.string(), &module_list));

  ASSERT_EQ(module_list->NumberOfModules, 3u);

  for (size_t index = 0; index < module_list->NumberOfModules; ++index) {
    SCOPED_TRACE(base::StringPrintf("index %" PRIuS, index));
    ASSERT_NO_FATAL_FAILURE(ExpectModule(&expect_modules[index],
                                         &module_list->Modules[index],
                                         string_file.string(),
                                         module_paths[index],
                                         module_pdbs[index],
                                         &uuids[index],
                                         0,
                                         ages[index],
                                         nullptr,
                                         0,
                                         false));
  }
}

TEST(MinidumpModuleWriterDeathTest, NoModuleName) {
  MinidumpFileWriter minidump_file_writer;
  auto module_list_writer = std::make_unique<MinidumpModuleListWriter>();
  auto module_writer = std::make_unique<MinidumpModuleWriter>();
  module_list_writer->AddModule(std::move(module_writer));
  ASSERT_TRUE(minidump_file_writer.AddStream(std::move(module_list_writer)));

  StringFile string_file;
  ASSERT_DEATH_CHECK(minidump_file_writer.WriteEverything(&string_file),
                     "name_");
}

}  // namespace
}  // namespace test
}  // namespace crashpad
