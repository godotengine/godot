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

#include "minidump/minidump_crashpad_info_writer.h"

#include <windows.h>
#include <dbghelp.h>

#include <map>
#include <string>
#include <utility>

#include "gtest/gtest.h"
#include "minidump/minidump_extensions.h"
#include "minidump/minidump_file_writer.h"
#include "minidump/minidump_module_crashpad_info_writer.h"
#include "minidump/minidump_simple_string_dictionary_writer.h"
#include "minidump/test/minidump_file_writer_test_util.h"
#include "minidump/test/minidump_string_writer_test_util.h"
#include "minidump/test/minidump_writable_test_util.h"
#include "snapshot/test/test_module_snapshot.h"
#include "snapshot/test/test_process_snapshot.h"
#include "util/file/string_file.h"

namespace crashpad {
namespace test {
namespace {

void GetCrashpadInfoStream(
    const std::string& file_contents,
    const MinidumpCrashpadInfo** crashpad_info,
    const MinidumpSimpleStringDictionary** simple_annotations,
    const MinidumpModuleCrashpadInfoList** module_list) {
  const MINIDUMP_DIRECTORY* directory;
  const MINIDUMP_HEADER* header =
      MinidumpHeaderAtStart(file_contents, &directory);
  ASSERT_NO_FATAL_FAILURE(VerifyMinidumpHeader(header, 1, 0));
  ASSERT_TRUE(directory);

  ASSERT_EQ(directory[0].StreamType, kMinidumpStreamTypeCrashpadInfo);

  *crashpad_info = MinidumpWritableAtLocationDescriptor<MinidumpCrashpadInfo>(
      file_contents, directory[0].Location);
  ASSERT_TRUE(*crashpad_info);

  *simple_annotations =
      MinidumpWritableAtLocationDescriptor<MinidumpSimpleStringDictionary>(
          file_contents, (*crashpad_info)->simple_annotations);

  *module_list =
      MinidumpWritableAtLocationDescriptor<MinidumpModuleCrashpadInfoList>(
          file_contents, (*crashpad_info)->module_list);
}

TEST(MinidumpCrashpadInfoWriter, Empty) {
  MinidumpFileWriter minidump_file_writer;
  auto crashpad_info_writer = std::make_unique<MinidumpCrashpadInfoWriter>();
  EXPECT_FALSE(crashpad_info_writer->IsUseful());

  ASSERT_TRUE(minidump_file_writer.AddStream(std::move(crashpad_info_writer)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  const MinidumpCrashpadInfo* crashpad_info = nullptr;
  const MinidumpSimpleStringDictionary* simple_annotations = nullptr;
  const MinidumpModuleCrashpadInfoList* module_list = nullptr;

  ASSERT_NO_FATAL_FAILURE(GetCrashpadInfoStream(
      string_file.string(), &crashpad_info, &simple_annotations, &module_list));

  EXPECT_EQ(crashpad_info->version, MinidumpCrashpadInfo::kVersion);
  EXPECT_EQ(crashpad_info->report_id, UUID());
  EXPECT_EQ(crashpad_info->client_id, UUID());
  EXPECT_FALSE(simple_annotations);
  EXPECT_FALSE(module_list);
}

TEST(MinidumpCrashpadInfoWriter, ReportAndClientID) {
  MinidumpFileWriter minidump_file_writer;
  auto crashpad_info_writer = std::make_unique<MinidumpCrashpadInfoWriter>();

  UUID report_id;
  ASSERT_TRUE(
      report_id.InitializeFromString("01234567-89ab-cdef-0123-456789abcdef"));
  crashpad_info_writer->SetReportID(report_id);

  UUID client_id;
  ASSERT_TRUE(
      client_id.InitializeFromString("00112233-4455-6677-8899-aabbccddeeff"));
  crashpad_info_writer->SetClientID(client_id);

  EXPECT_TRUE(crashpad_info_writer->IsUseful());

  ASSERT_TRUE(minidump_file_writer.AddStream(std::move(crashpad_info_writer)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  const MinidumpCrashpadInfo* crashpad_info = nullptr;
  const MinidumpSimpleStringDictionary* simple_annotations = nullptr;
  const MinidumpModuleCrashpadInfoList* module_list = nullptr;

  ASSERT_NO_FATAL_FAILURE(GetCrashpadInfoStream(
      string_file.string(), &crashpad_info, &simple_annotations, &module_list));

  EXPECT_EQ(crashpad_info->version, MinidumpCrashpadInfo::kVersion);
  EXPECT_EQ(crashpad_info->report_id, report_id);
  EXPECT_EQ(crashpad_info->client_id, client_id);
  EXPECT_FALSE(simple_annotations);
  EXPECT_FALSE(module_list);
}

TEST(MinidumpCrashpadInfoWriter, SimpleAnnotations) {
  MinidumpFileWriter minidump_file_writer;
  auto crashpad_info_writer = std::make_unique<MinidumpCrashpadInfoWriter>();

  static constexpr char kKey[] =
      "a thing that provides a means of gaining access to or understanding "
      "something";
  static constexpr char kValue[] =
      "the numerical amount denoted by an algebraic term; a magnitude, "
      "quantity, or number";
  auto simple_string_dictionary_writer =
      std::make_unique<MinidumpSimpleStringDictionaryWriter>();
  auto simple_string_dictionary_entry_writer =
      std::make_unique<MinidumpSimpleStringDictionaryEntryWriter>();
  simple_string_dictionary_entry_writer->SetKeyValue(kKey, kValue);
  simple_string_dictionary_writer->AddEntry(
      std::move(simple_string_dictionary_entry_writer));
  crashpad_info_writer->SetSimpleAnnotations(
      std::move(simple_string_dictionary_writer));

  EXPECT_TRUE(crashpad_info_writer->IsUseful());

  ASSERT_TRUE(minidump_file_writer.AddStream(std::move(crashpad_info_writer)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  const MinidumpCrashpadInfo* crashpad_info = nullptr;
  const MinidumpSimpleStringDictionary* simple_annotations = nullptr;
  const MinidumpModuleCrashpadInfoList* module_list = nullptr;

  ASSERT_NO_FATAL_FAILURE(GetCrashpadInfoStream(
      string_file.string(), &crashpad_info, &simple_annotations, &module_list));

  EXPECT_EQ(crashpad_info->version, MinidumpCrashpadInfo::kVersion);
  EXPECT_FALSE(module_list);

  ASSERT_TRUE(simple_annotations);
  ASSERT_EQ(simple_annotations->count, 1u);
  EXPECT_EQ(MinidumpUTF8StringAtRVAAsString(string_file.string(),
                                            simple_annotations->entries[0].key),
            kKey);
  EXPECT_EQ(MinidumpUTF8StringAtRVAAsString(
                string_file.string(), simple_annotations->entries[0].value),
            kValue);
}

TEST(MinidumpCrashpadInfoWriter, CrashpadModuleList) {
  constexpr uint32_t kMinidumpModuleListIndex = 3;

  MinidumpFileWriter minidump_file_writer;
  auto crashpad_info_writer = std::make_unique<MinidumpCrashpadInfoWriter>();

  auto module_list_writer =
      std::make_unique<MinidumpModuleCrashpadInfoListWriter>();
  auto module_writer = std::make_unique<MinidumpModuleCrashpadInfoWriter>();
  module_list_writer->AddModule(std::move(module_writer),
                                kMinidumpModuleListIndex);
  crashpad_info_writer->SetModuleList(std::move(module_list_writer));

  EXPECT_TRUE(crashpad_info_writer->IsUseful());

  ASSERT_TRUE(minidump_file_writer.AddStream(std::move(crashpad_info_writer)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  const MinidumpCrashpadInfo* crashpad_info = nullptr;
  const MinidumpSimpleStringDictionary* simple_annotations = nullptr;
  const MinidumpModuleCrashpadInfoList* module_list = nullptr;

  ASSERT_NO_FATAL_FAILURE(GetCrashpadInfoStream(
      string_file.string(), &crashpad_info, &simple_annotations, &module_list));

  EXPECT_EQ(crashpad_info->version, MinidumpCrashpadInfo::kVersion);
  EXPECT_FALSE(simple_annotations);

  ASSERT_TRUE(module_list);
  ASSERT_EQ(module_list->count, 1u);

  EXPECT_EQ(module_list->modules[0].minidump_module_list_index,
            kMinidumpModuleListIndex);
  const MinidumpModuleCrashpadInfo* module =
      MinidumpWritableAtLocationDescriptor<MinidumpModuleCrashpadInfo>(
          string_file.string(), module_list->modules[0].location);
  ASSERT_TRUE(module);

  EXPECT_EQ(module->version, MinidumpModuleCrashpadInfo::kVersion);
  EXPECT_EQ(module->list_annotations.DataSize, 0u);
  EXPECT_EQ(module->list_annotations.Rva, 0u);
  EXPECT_EQ(module->simple_annotations.DataSize, 0u);
  EXPECT_EQ(module->simple_annotations.Rva, 0u);
}

TEST(MinidumpCrashpadInfoWriter, InitializeFromSnapshot) {
  UUID report_id;
  ASSERT_TRUE(
      report_id.InitializeFromString("fedcba98-7654-3210-fedc-ba9876543210"));

  UUID client_id;
  ASSERT_TRUE(
      client_id.InitializeFromString("fedcba98-7654-3210-0123-456789abcdef"));

  static constexpr char kKey[] = "version";
  static constexpr char kValue[] = "40.0.2214.111";
  static constexpr char kEntry[] = "This is a simple annotation in a list.";

  // Test with a useless module, one that doesn’t carry anything that would
  // require MinidumpCrashpadInfo or any child object.
  auto process_snapshot = std::make_unique<TestProcessSnapshot>();

  auto module_snapshot = std::make_unique<TestModuleSnapshot>();
  process_snapshot->AddModule(std::move(module_snapshot));

  auto info_writer = std::make_unique<MinidumpCrashpadInfoWriter>();
  info_writer->InitializeFromSnapshot(process_snapshot.get());
  EXPECT_FALSE(info_writer->IsUseful());

  // Try again with a useful module.
  process_snapshot.reset(new TestProcessSnapshot());

  process_snapshot->SetReportID(report_id);
  process_snapshot->SetClientID(client_id);

  std::map<std::string, std::string> annotations_simple_map;
  annotations_simple_map[kKey] = kValue;
  process_snapshot->SetAnnotationsSimpleMap(annotations_simple_map);

  module_snapshot.reset(new TestModuleSnapshot());
  std::vector<std::string> annotations_list(1, std::string(kEntry));
  module_snapshot->SetAnnotationsVector(annotations_list);
  process_snapshot->AddModule(std::move(module_snapshot));

  info_writer.reset(new MinidumpCrashpadInfoWriter());
  info_writer->InitializeFromSnapshot(process_snapshot.get());
  EXPECT_TRUE(info_writer->IsUseful());

  MinidumpFileWriter minidump_file_writer;
  ASSERT_TRUE(minidump_file_writer.AddStream(std::move(info_writer)));

  StringFile string_file;
  ASSERT_TRUE(minidump_file_writer.WriteEverything(&string_file));

  const MinidumpCrashpadInfo* info = nullptr;
  const MinidumpSimpleStringDictionary* simple_annotations;
  const MinidumpModuleCrashpadInfoList* module_list;
  ASSERT_NO_FATAL_FAILURE(GetCrashpadInfoStream(
      string_file.string(), &info, &simple_annotations, &module_list));

  EXPECT_EQ(info->version, MinidumpCrashpadInfo::kVersion);

  EXPECT_EQ(info->report_id, report_id);
  EXPECT_EQ(info->client_id, client_id);

  ASSERT_TRUE(simple_annotations);
  ASSERT_EQ(simple_annotations->count, 1u);
  EXPECT_EQ(MinidumpUTF8StringAtRVAAsString(string_file.string(),
                                            simple_annotations->entries[0].key),
            kKey);
  EXPECT_EQ(MinidumpUTF8StringAtRVAAsString(
                string_file.string(), simple_annotations->entries[0].value),
            kValue);

  ASSERT_TRUE(module_list);
  ASSERT_EQ(module_list->count, 1u);

  EXPECT_EQ(module_list->modules[0].minidump_module_list_index, 0u);
  const MinidumpModuleCrashpadInfo* module =
      MinidumpWritableAtLocationDescriptor<MinidumpModuleCrashpadInfo>(
          string_file.string(), module_list->modules[0].location);
  ASSERT_TRUE(module);

  EXPECT_EQ(module->version, MinidumpModuleCrashpadInfo::kVersion);

  const MinidumpRVAList* list_annotations =
      MinidumpWritableAtLocationDescriptor<MinidumpRVAList>(
          string_file.string(), module->list_annotations);
  ASSERT_TRUE(list_annotations);

  ASSERT_EQ(list_annotations->count, 1u);
  EXPECT_EQ(MinidumpUTF8StringAtRVAAsString(string_file.string(),
                                            list_annotations->children[0]),
            kEntry);

  const MinidumpSimpleStringDictionary* module_simple_annotations =
      MinidumpWritableAtLocationDescriptor<MinidumpSimpleStringDictionary>(
          string_file.string(), module->simple_annotations);
  EXPECT_FALSE(module_simple_annotations);
}

}  // namespace
}  // namespace test
}  // namespace crashpad
