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

#include "minidump/minidump_simple_string_dictionary_writer.h"

#include <stdint.h>

#include <utility>

#include "gtest/gtest.h"
#include "minidump/test/minidump_string_writer_test_util.h"
#include "minidump/test/minidump_writable_test_util.h"
#include "util/file/string_file.h"

namespace crashpad {
namespace test {
namespace {

const MinidumpSimpleStringDictionary* MinidumpSimpleStringDictionaryAtStart(
    const std::string& file_contents,
    size_t count) {
  MINIDUMP_LOCATION_DESCRIPTOR location_descriptor;
  location_descriptor.DataSize = static_cast<uint32_t>(
      sizeof(MinidumpSimpleStringDictionary) +
      count * sizeof(MinidumpSimpleStringDictionaryEntry));
  location_descriptor.Rva = 0;
  return MinidumpWritableAtLocationDescriptor<MinidumpSimpleStringDictionary>(
      file_contents, location_descriptor);
}

TEST(MinidumpSimpleStringDictionaryWriter, EmptySimpleStringDictionary) {
  StringFile string_file;

  MinidumpSimpleStringDictionaryWriter dictionary_writer;

  EXPECT_FALSE(dictionary_writer.IsUseful());

  EXPECT_TRUE(dictionary_writer.WriteEverything(&string_file));
  ASSERT_EQ(string_file.string().size(),
            sizeof(MinidumpSimpleStringDictionary));

  const MinidumpSimpleStringDictionary* dictionary =
      MinidumpSimpleStringDictionaryAtStart(string_file.string(), 0);
  ASSERT_TRUE(dictionary);
  EXPECT_EQ(dictionary->count, 0u);
}

TEST(MinidumpSimpleStringDictionaryWriter, EmptyKeyValue) {
  StringFile string_file;

  MinidumpSimpleStringDictionaryWriter dictionary_writer;
  auto entry_writer =
      std::make_unique<MinidumpSimpleStringDictionaryEntryWriter>();
  dictionary_writer.AddEntry(std::move(entry_writer));

  EXPECT_TRUE(dictionary_writer.IsUseful());

  EXPECT_TRUE(dictionary_writer.WriteEverything(&string_file));
  ASSERT_EQ(string_file.string().size(),
            sizeof(MinidumpSimpleStringDictionary) +
                sizeof(MinidumpSimpleStringDictionaryEntry) +
                2 * sizeof(MinidumpUTF8String) + 1 + 3 + 1);  // 3 for padding

  const MinidumpSimpleStringDictionary* dictionary =
      MinidumpSimpleStringDictionaryAtStart(string_file.string(), 1);
  ASSERT_TRUE(dictionary);
  EXPECT_EQ(dictionary->count, 1u);
  EXPECT_EQ(dictionary->entries[0].key, 12u);
  EXPECT_EQ(dictionary->entries[0].value, 20u);
  EXPECT_EQ(MinidumpUTF8StringAtRVAAsString(string_file.string(),
                                            dictionary->entries[0].key),
            "");
  EXPECT_EQ(MinidumpUTF8StringAtRVAAsString(string_file.string(),
                                            dictionary->entries[0].value),
            "");
}

TEST(MinidumpSimpleStringDictionaryWriter, OneKeyValue) {
  StringFile string_file;

  char kKey[] = "key";
  char kValue[] = "value";

  MinidumpSimpleStringDictionaryWriter dictionary_writer;
  auto entry_writer =
      std::make_unique<MinidumpSimpleStringDictionaryEntryWriter>();
  entry_writer->SetKeyValue(kKey, kValue);
  dictionary_writer.AddEntry(std::move(entry_writer));

  EXPECT_TRUE(dictionary_writer.IsUseful());

  EXPECT_TRUE(dictionary_writer.WriteEverything(&string_file));
  ASSERT_EQ(string_file.string().size(),
            sizeof(MinidumpSimpleStringDictionary) +
                sizeof(MinidumpSimpleStringDictionaryEntry) +
                2 * sizeof(MinidumpUTF8String) + sizeof(kKey) + sizeof(kValue));

  const MinidumpSimpleStringDictionary* dictionary =
      MinidumpSimpleStringDictionaryAtStart(string_file.string(), 1);
  ASSERT_TRUE(dictionary);
  EXPECT_EQ(dictionary->count, 1u);
  EXPECT_EQ(dictionary->entries[0].key, 12u);
  EXPECT_EQ(dictionary->entries[0].value, 20u);
  EXPECT_EQ(MinidumpUTF8StringAtRVAAsString(string_file.string(),
                                            dictionary->entries[0].key),
            kKey);
  EXPECT_EQ(MinidumpUTF8StringAtRVAAsString(string_file.string(),
                                            dictionary->entries[0].value),
            kValue);
}

TEST(MinidumpSimpleStringDictionaryWriter, ThreeKeysValues) {
  StringFile string_file;

  char kKey0[] = "m0";
  char kValue0[] = "value0";
  char kKey1[] = "zzz1";
  char kValue1[] = "v1";
  char kKey2[] = "aa2";
  char kValue2[] = "val2";

  MinidumpSimpleStringDictionaryWriter dictionary_writer;
  auto entry_writer_0 =
      std::make_unique<MinidumpSimpleStringDictionaryEntryWriter>();
  entry_writer_0->SetKeyValue(kKey0, kValue0);
  dictionary_writer.AddEntry(std::move(entry_writer_0));
  auto entry_writer_1 =
      std::make_unique<MinidumpSimpleStringDictionaryEntryWriter>();
  entry_writer_1->SetKeyValue(kKey1, kValue1);
  dictionary_writer.AddEntry(std::move(entry_writer_1));
  auto entry_writer_2 =
      std::make_unique<MinidumpSimpleStringDictionaryEntryWriter>();
  entry_writer_2->SetKeyValue(kKey2, kValue2);
  dictionary_writer.AddEntry(std::move(entry_writer_2));

  EXPECT_TRUE(dictionary_writer.IsUseful());

  EXPECT_TRUE(dictionary_writer.WriteEverything(&string_file));
  ASSERT_EQ(string_file.string().size(),
            sizeof(MinidumpSimpleStringDictionary) +
                3 * sizeof(MinidumpSimpleStringDictionaryEntry) +
                6 * sizeof(MinidumpUTF8String) + sizeof(kKey2) +
                sizeof(kValue2) + 3 + sizeof(kKey0) + 1 + sizeof(kValue0) + 1 +
                sizeof(kKey1) + 3 + sizeof(kValue1));

  const MinidumpSimpleStringDictionary* dictionary =
      MinidumpSimpleStringDictionaryAtStart(string_file.string(), 3);
  ASSERT_TRUE(dictionary);
  EXPECT_EQ(dictionary->count, 3u);
  EXPECT_EQ(dictionary->entries[0].key, 28u);
  EXPECT_EQ(dictionary->entries[0].value, 36u);
  EXPECT_EQ(dictionary->entries[1].key, 48u);
  EXPECT_EQ(dictionary->entries[1].value, 56u);
  EXPECT_EQ(dictionary->entries[2].key, 68u);
  EXPECT_EQ(dictionary->entries[2].value, 80u);

  // The entries don’t appear in the order they were added. The current
  // implementation uses a std::map and sorts keys, so the entires appear in
  // alphabetical order. However, this is an implementation detail, and it’s OK
  // if the writer stops sorting in this order. Testing for a specific order is
  // just the easiest way to write this test while the writer will output things
  // in a known order.
  EXPECT_EQ(MinidumpUTF8StringAtRVAAsString(string_file.string(),
                                            dictionary->entries[0].key),
            kKey2);
  EXPECT_EQ(MinidumpUTF8StringAtRVAAsString(string_file.string(),
                                            dictionary->entries[0].value),
            kValue2);
  EXPECT_EQ(MinidumpUTF8StringAtRVAAsString(string_file.string(),
                                            dictionary->entries[1].key),
            kKey0);
  EXPECT_EQ(MinidumpUTF8StringAtRVAAsString(string_file.string(),
                                            dictionary->entries[1].value),
            kValue0);
  EXPECT_EQ(MinidumpUTF8StringAtRVAAsString(string_file.string(),
                                            dictionary->entries[2].key),
            kKey1);
  EXPECT_EQ(MinidumpUTF8StringAtRVAAsString(string_file.string(),
                                            dictionary->entries[2].value),
            kValue1);
}

TEST(MinidumpSimpleStringDictionaryWriter, DuplicateKeyValue) {
  StringFile string_file;

  char kKey[] = "key";
  char kValue0[] = "fake_value";
  char kValue1[] = "value";

  MinidumpSimpleStringDictionaryWriter dictionary_writer;
  auto entry_writer_0 =
      std::make_unique<MinidumpSimpleStringDictionaryEntryWriter>();
  entry_writer_0->SetKeyValue(kKey, kValue0);
  dictionary_writer.AddEntry(std::move(entry_writer_0));
  auto entry_writer_1 =
      std::make_unique<MinidumpSimpleStringDictionaryEntryWriter>();
  entry_writer_1->SetKeyValue(kKey, kValue1);
  dictionary_writer.AddEntry(std::move(entry_writer_1));

  EXPECT_TRUE(dictionary_writer.IsUseful());

  EXPECT_TRUE(dictionary_writer.WriteEverything(&string_file));
  ASSERT_EQ(string_file.string().size(),
            sizeof(MinidumpSimpleStringDictionary) +
                sizeof(MinidumpSimpleStringDictionaryEntry) +
                2 * sizeof(MinidumpUTF8String) + sizeof(kKey) +
                sizeof(kValue1));

  const MinidumpSimpleStringDictionary* dictionary =
      MinidumpSimpleStringDictionaryAtStart(string_file.string(), 1);
  ASSERT_TRUE(dictionary);
  EXPECT_EQ(dictionary->count, 1u);
  EXPECT_EQ(dictionary->entries[0].key, 12u);
  EXPECT_EQ(dictionary->entries[0].value, 20u);
  EXPECT_EQ(MinidumpUTF8StringAtRVAAsString(string_file.string(),
                                            dictionary->entries[0].key),
            kKey);
  EXPECT_EQ(MinidumpUTF8StringAtRVAAsString(string_file.string(),
                                            dictionary->entries[0].value),
            kValue1);
}

TEST(MinidumpSimpleStringDictionaryWriter, InitializeFromMap) {
  char kKey0[] = "Dictionaries";
  char kValue0[] = "USEFUL*";
  char kKey1[] = "#1 Key!";
  char kValue1[] = "";
  char kKey2[] = "key two";
  char kValue2[] = "value two";

  std::map<std::string, std::string> map;
  map[kKey0] = kValue0;
  map[kKey1] = kValue1;
  map[kKey2] = kValue2;

  MinidumpSimpleStringDictionaryWriter dictionary_writer;
  dictionary_writer.InitializeFromMap(map);

  EXPECT_TRUE(dictionary_writer.IsUseful());

  StringFile string_file;
  ASSERT_TRUE(dictionary_writer.WriteEverything(&string_file));

  const MinidumpSimpleStringDictionary* dictionary =
      MinidumpSimpleStringDictionaryAtStart(string_file.string(), map.size());
  ASSERT_TRUE(dictionary);
  ASSERT_EQ(dictionary->count, 3u);

  // The entries don’t appear in the order they were added. The current
  // implementation uses a std::map and sorts keys, so the entires appear in
  // alphabetical order. However, this is an implementation detail, and it’s OK
  // if the writer stops sorting in this order. Testing for a specific order is
  // just the easiest way to write this test while the writer will output things
  // in a known order.
  EXPECT_EQ(MinidumpUTF8StringAtRVAAsString(string_file.string(),
                                            dictionary->entries[0].key),
            kKey1);
  EXPECT_EQ(MinidumpUTF8StringAtRVAAsString(string_file.string(),
                                            dictionary->entries[0].value),
            kValue1);
  EXPECT_EQ(MinidumpUTF8StringAtRVAAsString(string_file.string(),
                                            dictionary->entries[1].key),
            kKey0);
  EXPECT_EQ(MinidumpUTF8StringAtRVAAsString(string_file.string(),
                                            dictionary->entries[1].value),
            kValue0);
  EXPECT_EQ(MinidumpUTF8StringAtRVAAsString(string_file.string(),
                                            dictionary->entries[2].key),
            kKey2);
  EXPECT_EQ(MinidumpUTF8StringAtRVAAsString(string_file.string(),
                                            dictionary->entries[2].value),
            kValue2);
}

}  // namespace
}  // namespace test
}  // namespace crashpad
