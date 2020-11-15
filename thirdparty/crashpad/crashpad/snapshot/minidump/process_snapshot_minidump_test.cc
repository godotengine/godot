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

#include "snapshot/minidump/process_snapshot_minidump.h"

#include <windows.h>
#include <dbghelp.h>
#include <string.h>

#include <memory>

#include "gtest/gtest.h"
#include "snapshot/minidump/minidump_annotation_reader.h"
#include "snapshot/module_snapshot.h"
#include "util/file/string_file.h"

namespace crashpad {
namespace test {
namespace {

TEST(ProcessSnapshotMinidump, EmptyFile) {
  StringFile string_file;
  ProcessSnapshotMinidump process_snapshot;

  EXPECT_FALSE(process_snapshot.Initialize(&string_file));
}

TEST(ProcessSnapshotMinidump, InvalidSignatureAndVersion) {
  StringFile string_file;

  MINIDUMP_HEADER header = {};

  EXPECT_TRUE(string_file.Write(&header, sizeof(header)));

  ProcessSnapshotMinidump process_snapshot;
  EXPECT_FALSE(process_snapshot.Initialize(&string_file));
}

TEST(ProcessSnapshotMinidump, Empty) {
  StringFile string_file;

  MINIDUMP_HEADER header = {};
  header.Signature = MINIDUMP_SIGNATURE;
  header.Version = MINIDUMP_VERSION;

  EXPECT_TRUE(string_file.Write(&header, sizeof(header)));

  ProcessSnapshotMinidump process_snapshot;
  EXPECT_TRUE(process_snapshot.Initialize(&string_file));

  UUID client_id;
  process_snapshot.ClientID(&client_id);
  EXPECT_EQ(client_id, UUID());

  EXPECT_TRUE(process_snapshot.AnnotationsSimpleMap().empty());
}

// Writes |string| to |writer| as a MinidumpUTF8String, and returns the file
// offset of the beginning of the string.
RVA WriteString(FileWriterInterface* writer, const std::string& string) {
  RVA rva = static_cast<RVA>(writer->SeekGet());

  uint32_t string_size = static_cast<uint32_t>(string.size());
  EXPECT_TRUE(writer->Write(&string_size, sizeof(string_size)));

  // Include the trailing NUL character.
  EXPECT_TRUE(writer->Write(string.c_str(), string.size() + 1));

  return rva;
}

// Writes |dictionary| to |writer| as a MinidumpSimpleStringDictionary, and
// populates |location| with a location descriptor identifying what was written.
void WriteMinidumpSimpleStringDictionary(
    MINIDUMP_LOCATION_DESCRIPTOR* location,
    FileWriterInterface* writer,
    const std::map<std::string, std::string>& dictionary) {
  std::vector<MinidumpSimpleStringDictionaryEntry> entries;
  for (const auto& it : dictionary) {
    MinidumpSimpleStringDictionaryEntry entry;
    entry.key = WriteString(writer, it.first);
    entry.value = WriteString(writer, it.second);
    entries.push_back(entry);
  }

  location->Rva = static_cast<RVA>(writer->SeekGet());

  const uint32_t simple_string_dictionary_entries =
      static_cast<uint32_t>(entries.size());
  EXPECT_TRUE(writer->Write(&simple_string_dictionary_entries,
                            sizeof(simple_string_dictionary_entries)));
  for (const MinidumpSimpleStringDictionaryEntry& entry : entries) {
    EXPECT_TRUE(writer->Write(&entry, sizeof(entry)));
  }

  location->DataSize = static_cast<uint32_t>(
      sizeof(simple_string_dictionary_entries) +
      entries.size() * sizeof(MinidumpSimpleStringDictionaryEntry));
}

// Writes |strings| to |writer| as a MinidumpRVAList referencing
// MinidumpUTF8String objects, and populates |location| with a location
// descriptor identifying what was written.
void WriteMinidumpStringList(MINIDUMP_LOCATION_DESCRIPTOR* location,
                             FileWriterInterface* writer,
                             const std::vector<std::string>& strings) {
  std::vector<RVA> rvas;
  for (const std::string& string : strings) {
    rvas.push_back(WriteString(writer, string));
  }

  location->Rva = static_cast<RVA>(writer->SeekGet());

  const uint32_t string_list_entries = static_cast<uint32_t>(rvas.size());
  EXPECT_TRUE(writer->Write(&string_list_entries, sizeof(string_list_entries)));
  for (RVA rva : rvas) {
    EXPECT_TRUE(writer->Write(&rva, sizeof(rva)));
  }

  location->DataSize = static_cast<uint32_t>(sizeof(string_list_entries) +
                                             rvas.size() * sizeof(RVA));
}

// Writes |data| to |writer| as a MinidumpByteArray, and returns the file offset
// from the beginning of the string.
RVA WriteByteArray(FileWriterInterface* writer,
                   const std::vector<uint8_t> data) {
  auto rva = static_cast<RVA>(writer->SeekGet());

  auto length = static_cast<uint32_t>(data.size());
  EXPECT_TRUE(writer->Write(&length, sizeof(length)));
  EXPECT_TRUE(writer->Write(data.data(), length));

  return rva;
}

// Writes |annotations| to |writer| as a MinidumpAnnotationList, and populates
// |location| with a location descriptor identifying what was written.
void WriteMinidumpAnnotationList(
    MINIDUMP_LOCATION_DESCRIPTOR* location,
    FileWriterInterface* writer,
    const std::vector<AnnotationSnapshot>& annotations) {
  std::vector<MinidumpAnnotation> minidump_annotations;
  for (const auto& it : annotations) {
    MinidumpAnnotation annotation;
    annotation.name = WriteString(writer, it.name);
    annotation.type = it.type;
    annotation.reserved = 0;
    annotation.value = WriteByteArray(writer, it.value);
    minidump_annotations.push_back(annotation);
  }

  location->Rva = static_cast<RVA>(writer->SeekGet());

  auto count = static_cast<uint32_t>(minidump_annotations.size());
  EXPECT_TRUE(writer->Write(&count, sizeof(count)));

  for (const auto& it : minidump_annotations) {
    EXPECT_TRUE(writer->Write(&it, sizeof(MinidumpAnnotation)));
  }

  location->DataSize =
      sizeof(MinidumpAnnotationList) + count * sizeof(MinidumpAnnotation);
}

TEST(ProcessSnapshotMinidump, ClientID) {
  StringFile string_file;

  MINIDUMP_HEADER header = {};
  EXPECT_TRUE(string_file.Write(&header, sizeof(header)));

  UUID client_id;
  ASSERT_TRUE(
      client_id.InitializeFromString("0001f4a9-d00d-5155-0a55-c0ffeec0ffee"));

  MinidumpCrashpadInfo crashpad_info = {};
  crashpad_info.version = MinidumpCrashpadInfo::kVersion;
  crashpad_info.client_id = client_id;

  MINIDUMP_DIRECTORY crashpad_info_directory = {};
  crashpad_info_directory.StreamType = kMinidumpStreamTypeCrashpadInfo;
  crashpad_info_directory.Location.Rva =
      static_cast<RVA>(string_file.SeekGet());
  EXPECT_TRUE(string_file.Write(&crashpad_info, sizeof(crashpad_info)));
  crashpad_info_directory.Location.DataSize = sizeof(crashpad_info);

  header.StreamDirectoryRva = static_cast<RVA>(string_file.SeekGet());
  EXPECT_TRUE(string_file.Write(&crashpad_info_directory,
                                sizeof(crashpad_info_directory)));

  header.Signature = MINIDUMP_SIGNATURE;
  header.Version = MINIDUMP_VERSION;
  header.NumberOfStreams = 1;
  EXPECT_TRUE(string_file.SeekSet(0));
  EXPECT_TRUE(string_file.Write(&header, sizeof(header)));

  ProcessSnapshotMinidump process_snapshot;
  EXPECT_TRUE(process_snapshot.Initialize(&string_file));

  UUID actual_client_id;
  process_snapshot.ClientID(&actual_client_id);
  EXPECT_EQ(actual_client_id, client_id);

  EXPECT_TRUE(process_snapshot.AnnotationsSimpleMap().empty());
}

TEST(ProcessSnapshotMinidump, AnnotationsSimpleMap) {
  StringFile string_file;

  MINIDUMP_HEADER header = {};
  EXPECT_TRUE(string_file.Write(&header, sizeof(header)));

  MinidumpCrashpadInfo crashpad_info = {};
  crashpad_info.version = MinidumpCrashpadInfo::kVersion;

  std::map<std::string, std::string> dictionary;
  dictionary["the first key"] = "THE FIRST VALUE EVER!";
  dictionary["2key"] = "a lowly second value";
  WriteMinidumpSimpleStringDictionary(
      &crashpad_info.simple_annotations, &string_file, dictionary);

  MINIDUMP_DIRECTORY crashpad_info_directory = {};
  crashpad_info_directory.StreamType = kMinidumpStreamTypeCrashpadInfo;
  crashpad_info_directory.Location.Rva =
      static_cast<RVA>(string_file.SeekGet());
  EXPECT_TRUE(string_file.Write(&crashpad_info, sizeof(crashpad_info)));
  crashpad_info_directory.Location.DataSize = sizeof(crashpad_info);

  header.StreamDirectoryRva = static_cast<RVA>(string_file.SeekGet());
  EXPECT_TRUE(string_file.Write(&crashpad_info_directory,
                                sizeof(crashpad_info_directory)));

  header.Signature = MINIDUMP_SIGNATURE;
  header.Version = MINIDUMP_VERSION;
  header.NumberOfStreams = 1;
  EXPECT_TRUE(string_file.SeekSet(0));
  EXPECT_TRUE(string_file.Write(&header, sizeof(header)));

  ProcessSnapshotMinidump process_snapshot;
  EXPECT_TRUE(process_snapshot.Initialize(&string_file));

  UUID client_id;
  process_snapshot.ClientID(&client_id);
  EXPECT_EQ(client_id, UUID());

  const auto annotations_simple_map = process_snapshot.AnnotationsSimpleMap();
  EXPECT_EQ(annotations_simple_map, dictionary);
}

TEST(ProcessSnapshotMinidump, AnnotationObjects) {
  StringFile string_file;

  MINIDUMP_HEADER header{};
  EXPECT_TRUE(string_file.Write(&header, sizeof(header)));

  std::vector<AnnotationSnapshot> annotations;
  annotations.emplace_back(
      AnnotationSnapshot("name 1", 0xBBBB, {'t', 'e', '\0', 's', 't', '\0'}));
  annotations.emplace_back(
      AnnotationSnapshot("name 2", 0xABBA, {0xF0, 0x9F, 0x92, 0x83}));

  MINIDUMP_LOCATION_DESCRIPTOR location;
  WriteMinidumpAnnotationList(&location, &string_file, annotations);

  std::vector<AnnotationSnapshot> read_annotations;
  EXPECT_TRUE(internal::ReadMinidumpAnnotationList(
      &string_file, location, &read_annotations));

  EXPECT_EQ(read_annotations, annotations);
}

TEST(ProcessSnapshotMinidump, Modules) {
  StringFile string_file;

  MINIDUMP_HEADER header = {};
  EXPECT_TRUE(string_file.Write(&header, sizeof(header)));

  MINIDUMP_MODULE minidump_module = {};
  uint32_t minidump_module_count = 4;

  MINIDUMP_DIRECTORY minidump_module_list_directory = {};
  minidump_module_list_directory.StreamType = kMinidumpStreamTypeModuleList;
  minidump_module_list_directory.Location.DataSize =
      sizeof(MINIDUMP_MODULE_LIST) +
      minidump_module_count * sizeof(MINIDUMP_MODULE);
  minidump_module_list_directory.Location.Rva =
      static_cast<RVA>(string_file.SeekGet());

  EXPECT_TRUE(
      string_file.Write(&minidump_module_count, sizeof(minidump_module_count)));
  for (uint32_t minidump_module_index = 0;
       minidump_module_index < minidump_module_count;
       ++minidump_module_index) {
    EXPECT_TRUE(string_file.Write(&minidump_module, sizeof(minidump_module)));
  }

  MinidumpModuleCrashpadInfo crashpad_module_0 = {};
  crashpad_module_0.version = MinidumpModuleCrashpadInfo::kVersion;
  std::map<std::string, std::string> dictionary_0;
  dictionary_0["ptype"] = "browser";
  dictionary_0["pid"] = "12345";
  WriteMinidumpSimpleStringDictionary(
      &crashpad_module_0.simple_annotations, &string_file, dictionary_0);

  MinidumpModuleCrashpadInfoLink crashpad_module_0_link = {};
  crashpad_module_0_link.minidump_module_list_index = 0;
  crashpad_module_0_link.location.DataSize = sizeof(crashpad_module_0);
  crashpad_module_0_link.location.Rva = static_cast<RVA>(string_file.SeekGet());
  EXPECT_TRUE(string_file.Write(&crashpad_module_0, sizeof(crashpad_module_0)));

  MinidumpModuleCrashpadInfo crashpad_module_2 = {};
  crashpad_module_2.version = MinidumpModuleCrashpadInfo::kVersion;
  std::map<std::string, std::string> dictionary_2;
  dictionary_2["fakemodule"] = "yes";
  WriteMinidumpSimpleStringDictionary(
      &crashpad_module_2.simple_annotations, &string_file, dictionary_2);

  std::vector<std::string> list_annotations_2;
  list_annotations_2.push_back("first string");
  list_annotations_2.push_back("last string");
  WriteMinidumpStringList(
      &crashpad_module_2.list_annotations, &string_file, list_annotations_2);

  MinidumpModuleCrashpadInfoLink crashpad_module_2_link = {};
  crashpad_module_2_link.minidump_module_list_index = 2;
  crashpad_module_2_link.location.DataSize = sizeof(crashpad_module_2);
  crashpad_module_2_link.location.Rva = static_cast<RVA>(string_file.SeekGet());
  EXPECT_TRUE(string_file.Write(&crashpad_module_2, sizeof(crashpad_module_2)));

  MinidumpModuleCrashpadInfo crashpad_module_4 = {};
  crashpad_module_4.version = MinidumpModuleCrashpadInfo::kVersion;
  std::vector<AnnotationSnapshot> annotations_4{
      {"first one", 0xBADE, {'a', 'b', 'c'}},
      {"2", 0xEDD1, {0x11, 0x22, 0x33}},
      {"threeeeee", 0xDADA, {'f'}},
  };
  WriteMinidumpAnnotationList(
      &crashpad_module_4.annotation_objects, &string_file, annotations_4);

  MinidumpModuleCrashpadInfoLink crashpad_module_4_link = {};
  crashpad_module_4_link.minidump_module_list_index = 3;
  crashpad_module_4_link.location.DataSize = sizeof(crashpad_module_4);
  crashpad_module_4_link.location.Rva = static_cast<RVA>(string_file.SeekGet());
  EXPECT_TRUE(string_file.Write(&crashpad_module_4, sizeof(crashpad_module_4)));

  MinidumpCrashpadInfo crashpad_info = {};
  crashpad_info.version = MinidumpCrashpadInfo::kVersion;

  uint32_t crashpad_module_count = 3;

  crashpad_info.module_list.DataSize =
      sizeof(MinidumpModuleCrashpadInfoList) +
      crashpad_module_count * sizeof(MinidumpModuleCrashpadInfoLink);
  crashpad_info.module_list.Rva = static_cast<RVA>(string_file.SeekGet());

  EXPECT_TRUE(
      string_file.Write(&crashpad_module_count, sizeof(crashpad_module_count)));
  EXPECT_TRUE(string_file.Write(&crashpad_module_0_link,
                                sizeof(crashpad_module_0_link)));
  EXPECT_TRUE(string_file.Write(&crashpad_module_2_link,
                                sizeof(crashpad_module_2_link)));
  EXPECT_TRUE(string_file.Write(&crashpad_module_4_link,
                                sizeof(crashpad_module_4_link)));

  MINIDUMP_DIRECTORY crashpad_info_directory = {};
  crashpad_info_directory.StreamType = kMinidumpStreamTypeCrashpadInfo;
  crashpad_info_directory.Location.DataSize = sizeof(crashpad_info);
  crashpad_info_directory.Location.Rva =
      static_cast<RVA>(string_file.SeekGet());
  EXPECT_TRUE(string_file.Write(&crashpad_info, sizeof(crashpad_info)));

  header.StreamDirectoryRva = static_cast<RVA>(string_file.SeekGet());
  EXPECT_TRUE(string_file.Write(&minidump_module_list_directory,
                                sizeof(minidump_module_list_directory)));
  EXPECT_TRUE(string_file.Write(&crashpad_info_directory,
                                sizeof(crashpad_info_directory)));

  header.Signature = MINIDUMP_SIGNATURE;
  header.Version = MINIDUMP_VERSION;
  header.NumberOfStreams = 2;
  EXPECT_TRUE(string_file.SeekSet(0));
  EXPECT_TRUE(string_file.Write(&header, sizeof(header)));

  ProcessSnapshotMinidump process_snapshot;
  EXPECT_TRUE(process_snapshot.Initialize(&string_file));

  std::vector<const ModuleSnapshot*> modules = process_snapshot.Modules();
  ASSERT_EQ(modules.size(), minidump_module_count);

  auto annotations_simple_map = modules[0]->AnnotationsSimpleMap();
  EXPECT_EQ(annotations_simple_map, dictionary_0);

  auto annotations_vector = modules[0]->AnnotationsVector();
  EXPECT_TRUE(annotations_vector.empty());

  annotations_simple_map = modules[1]->AnnotationsSimpleMap();
  EXPECT_TRUE(annotations_simple_map.empty());

  annotations_vector = modules[1]->AnnotationsVector();
  EXPECT_TRUE(annotations_vector.empty());

  annotations_simple_map = modules[2]->AnnotationsSimpleMap();
  EXPECT_EQ(annotations_simple_map, dictionary_2);

  annotations_vector = modules[2]->AnnotationsVector();
  EXPECT_EQ(annotations_vector, list_annotations_2);

  auto annotation_objects = modules[3]->AnnotationObjects();
  EXPECT_EQ(annotation_objects, annotations_4);
}

TEST(ProcessSnapshotMinidump, ProcessID) {
  StringFile string_file;

  MINIDUMP_HEADER header = {};
  ASSERT_TRUE(string_file.Write(&header, sizeof(header)));

  static const pid_t kTestProcessId = 42;
  MINIDUMP_MISC_INFO misc_info = {};
  misc_info.SizeOfInfo = sizeof(misc_info);
  misc_info.Flags1 = MINIDUMP_MISC1_PROCESS_ID;
  misc_info.ProcessId = kTestProcessId;

  MINIDUMP_DIRECTORY misc_directory = {};
  misc_directory.StreamType = kMinidumpStreamTypeMiscInfo;
  misc_directory.Location.DataSize = sizeof(misc_info);
  misc_directory.Location.Rva = static_cast<RVA>(string_file.SeekGet());
  ASSERT_TRUE(string_file.Write(&misc_info, sizeof(misc_info)));

  header.StreamDirectoryRva = static_cast<RVA>(string_file.SeekGet());
  ASSERT_TRUE(string_file.Write(&misc_directory, sizeof(misc_directory)));

  header.Signature = MINIDUMP_SIGNATURE;
  header.Version = MINIDUMP_VERSION;
  header.NumberOfStreams = 1;
  ASSERT_TRUE(string_file.SeekSet(0));
  ASSERT_TRUE(string_file.Write(&header, sizeof(header)));

  ProcessSnapshotMinidump process_snapshot;
  ASSERT_TRUE(process_snapshot.Initialize(&string_file));
  EXPECT_EQ(process_snapshot.ProcessID(), kTestProcessId);
}

}  // namespace
}  // namespace test
}  // namespace crashpad
