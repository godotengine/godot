// Copyright 2016 The Crashpad Authors. All rights reserved.
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

#include "minidump/minidump_unloaded_module_writer.h"

#include "base/strings/utf_string_conversions.h"
#include "gtest/gtest.h"
#include "minidump/minidump_file_writer.h"
#include "minidump/test/minidump_file_writer_test_util.h"
#include "minidump/test/minidump_string_writer_test_util.h"
#include "minidump/test/minidump_writable_test_util.h"
#include "util/file/string_file.h"

namespace crashpad {
namespace test {
namespace {

void ExpectUnloadedModule(const MINIDUMP_UNLOADED_MODULE* expected,
                          const MINIDUMP_UNLOADED_MODULE* observed,
                          const std::string& file_contents,
                          const std::string& expected_module_name) {
  EXPECT_EQ(observed->BaseOfImage, expected->BaseOfImage);
  EXPECT_EQ(observed->SizeOfImage, expected->SizeOfImage);
  EXPECT_EQ(observed->CheckSum, expected->CheckSum);
  EXPECT_EQ(observed->TimeDateStamp, expected->TimeDateStamp);
  EXPECT_NE(observed->ModuleNameRva, 0u);
  base::string16 observed_module_name_utf16 =
      MinidumpStringAtRVAAsString(file_contents, observed->ModuleNameRva);
  base::string16 expected_module_name_utf16 =
      base::UTF8ToUTF16(expected_module_name);
  EXPECT_EQ(observed_module_name_utf16, expected_module_name_utf16);
}

void GetUnloadedModuleListStream(
    const std::string& file_contents,
    const MINIDUMP_UNLOADED_MODULE_LIST** unloaded_module_list) {
  constexpr size_t kDirectoryOffset = sizeof(MINIDUMP_HEADER);
  constexpr size_t kUnloadedModuleListStreamOffset =
      kDirectoryOffset + sizeof(MINIDUMP_DIRECTORY);
  constexpr size_t kUnloadedModulesOffset =
      kUnloadedModuleListStreamOffset + sizeof(MINIDUMP_UNLOADED_MODULE_LIST);

  ASSERT_GE(file_contents.size(), kUnloadedModulesOffset);

  const MINIDUMP_DIRECTORY* directory;
  const MINIDUMP_HEADER* header =
      MinidumpHeaderAtStart(file_contents, &directory);
  ASSERT_NO_FATAL_FAILURE(VerifyMinidumpHeader(header, 1, 0));
  ASSERT_TRUE(directory);

  ASSERT_EQ(directory[0].StreamType, kMinidumpStreamTypeUnloadedModuleList);
  EXPECT_EQ(directory[0].Location.Rva, kUnloadedModuleListStreamOffset);

  *unloaded_module_list =
      MinidumpWritableAtLocationDescriptor<MINIDUMP_UNLOADED_MODULE_LIST>(
          file_contents, directory[0].Location);
  ASSERT_TRUE(unloaded_module_list);
}

TEST(MinidumpUnloadedModuleWriter, EmptyModule) {
  MinidumpFileWriter minidump_file_writer;
  auto unloaded_module_list_writer =
      std::make_unique<MinidumpUnloadedModuleListWriter>();

  static constexpr char kModuleName[] = "test_dll";

  auto unloaded_module_writer =
      std::make_unique<MinidumpUnloadedModuleWriter>();
  unloaded_module_writer->SetName(kModuleName);

  unloaded_module_list_writer->AddUnloadedModule(
      std::move(unloaded_module_writer));
  ASSERT_TRUE(
      minidump_file_writer.AddStream(std::move(unloaded_module_list_writer)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  ASSERT_GT(string_file.string().size(),
            sizeof(MINIDUMP_HEADER) + sizeof(MINIDUMP_DIRECTORY) +
                sizeof(MINIDUMP_UNLOADED_MODULE_LIST) +
                1 * sizeof(MINIDUMP_UNLOADED_MODULE));

  const MINIDUMP_UNLOADED_MODULE_LIST* unloaded_module_list = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      GetUnloadedModuleListStream(string_file.string(), &unloaded_module_list));

  EXPECT_EQ(unloaded_module_list->NumberOfEntries, 1u);

  MINIDUMP_UNLOADED_MODULE expected = {};
  ASSERT_NO_FATAL_FAILURE(
      ExpectUnloadedModule(&expected,
                           reinterpret_cast<const MINIDUMP_UNLOADED_MODULE*>(
                               &unloaded_module_list[1]),
                           string_file.string(),
                           kModuleName));
}

TEST(MinidumpUnloadedModuleWriter, OneModule) {
  MinidumpFileWriter minidump_file_writer;
  auto unloaded_module_list_writer =
      std::make_unique<MinidumpUnloadedModuleListWriter>();

  static constexpr char kModuleName[] = "statically_linked";
  constexpr uint64_t kModuleBase = 0x10da69000;
  constexpr uint32_t kModuleSize = 0x1000;
  constexpr uint32_t kChecksum = 0x76543210;
  constexpr time_t kTimestamp = 0x386d4380;

  auto unloaded_module_writer =
      std::make_unique<MinidumpUnloadedModuleWriter>();
  unloaded_module_writer->SetName(kModuleName);
  unloaded_module_writer->SetImageBaseAddress(kModuleBase);
  unloaded_module_writer->SetImageSize(kModuleSize);
  unloaded_module_writer->SetChecksum(kChecksum);
  unloaded_module_writer->SetTimestamp(kTimestamp);

  unloaded_module_list_writer->AddUnloadedModule(
      std::move(unloaded_module_writer));
  ASSERT_TRUE(
      minidump_file_writer.AddStream(std::move(unloaded_module_list_writer)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  ASSERT_GT(string_file.string().size(),
            sizeof(MINIDUMP_HEADER) + sizeof(MINIDUMP_DIRECTORY) +
                sizeof(MINIDUMP_UNLOADED_MODULE_LIST) +
                1 * sizeof(MINIDUMP_UNLOADED_MODULE));

  const MINIDUMP_UNLOADED_MODULE_LIST* unloaded_module_list = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      GetUnloadedModuleListStream(string_file.string(), &unloaded_module_list));

  EXPECT_EQ(unloaded_module_list->NumberOfEntries, 1u);

  MINIDUMP_UNLOADED_MODULE expected = {};
  expected.BaseOfImage = kModuleBase;
  expected.SizeOfImage = kModuleSize;
  expected.CheckSum = kChecksum;
  expected.TimeDateStamp = kTimestamp;

  ASSERT_NO_FATAL_FAILURE(
      ExpectUnloadedModule(&expected,
                           reinterpret_cast<const MINIDUMP_UNLOADED_MODULE*>(
                               &unloaded_module_list[1]),
                           string_file.string(),
                           kModuleName));
}

}  // namespace
}  // namespace test
}  // namespace crashpad
