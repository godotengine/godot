// Copyright 2017 The Crashpad Authors. All rights reserved.
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

#include "minidump/minidump_annotation_writer.h"

#include <memory>

#include "base/macros.h"
#include "gtest/gtest.h"
#include "minidump/minidump_extensions.h"
#include "minidump/test/minidump_byte_array_writer_test_util.h"
#include "minidump/test/minidump_string_writer_test_util.h"
#include "minidump/test/minidump_writable_test_util.h"
#include "util/file/string_file.h"

namespace crashpad {
namespace test {
namespace {

const MinidumpAnnotationList* MinidumpAnnotationListAtStart(
    const std::string& file_contents,
    uint32_t count) {
  MINIDUMP_LOCATION_DESCRIPTOR location_descriptor;
  location_descriptor.DataSize = static_cast<uint32_t>(
      sizeof(MinidumpAnnotationList) + count * sizeof(MinidumpAnnotation));
  location_descriptor.Rva = 0;
  return MinidumpWritableAtLocationDescriptor<MinidumpAnnotationList>(
      file_contents, location_descriptor);
}

TEST(MinidumpAnnotationWriter, EmptyList) {
  StringFile string_file;

  MinidumpAnnotationListWriter list_writer;

  EXPECT_FALSE(list_writer.IsUseful());

  EXPECT_TRUE(list_writer.WriteEverything(&string_file));

  ASSERT_EQ(string_file.string().size(), sizeof(MinidumpAnnotationList));

  auto* list = MinidumpAnnotationListAtStart(string_file.string(), 0);
  ASSERT_TRUE(list);
  EXPECT_EQ(list->count, 0u);
}

TEST(MinidumpAnnotationWriter, OneItem) {
  StringFile string_file;

  const char kName[] = "name";
  const uint16_t kType = 0xFFFF;
  const std::vector<uint8_t> kValue{'v', 'a', 'l', 'u', 'e', '\0'};

  auto annotation_writer = std::make_unique<MinidumpAnnotationWriter>();
  annotation_writer->InitializeWithData(kName, kType, kValue);

  MinidumpAnnotationListWriter list_writer;
  list_writer.AddObject(std::move(annotation_writer));

  EXPECT_TRUE(list_writer.IsUseful());

  EXPECT_TRUE(list_writer.WriteEverything(&string_file));

  ASSERT_EQ(string_file.string().size(),
            sizeof(MinidumpAnnotationList) + sizeof(MinidumpAnnotation) +
                sizeof(MinidumpUTF8String) + sizeof(kName) +
                sizeof(MinidumpByteArray) + kValue.size() +
                3);  // 3 for padding.

  auto* list = MinidumpAnnotationListAtStart(string_file.string(), 1);
  ASSERT_TRUE(list);
  EXPECT_EQ(list->count, 1u);
  EXPECT_EQ(MinidumpUTF8StringAtRVAAsString(string_file.string(),
                                            list->objects[0].name),
            kName);
  EXPECT_EQ(list->objects[0].type, kType);
  EXPECT_EQ(list->objects[0].reserved, 0u);
  EXPECT_EQ(

      MinidumpByteArrayAtRVA(string_file.string(), list->objects[0].value),
      kValue);
}

TEST(MinidumpAnnotationWriter, ThreeItems) {
  StringFile string_file;

  const char* kNames[] = {
      "~~FIRST~~", " second + ", "3",
  };
  const uint16_t kTypes[] = {
      0x1, 0xABCD, 0x42,
  };
  const std::vector<uint8_t> kValues[] = {
      {'\0'}, {0xB0, 0xA0, 0xD0, 0xD0, 0xD0}, {'T'},
  };

  MinidumpAnnotationListWriter list_writer;

  for (size_t i = 0; i < arraysize(kNames); ++i) {
    auto annotation = std::make_unique<MinidumpAnnotationWriter>();
    annotation->InitializeWithData(kNames[i], kTypes[i], kValues[i]);
    list_writer.AddObject(std::move(annotation));
  }

  EXPECT_TRUE(list_writer.WriteEverything(&string_file));

  ASSERT_EQ(string_file.string().size(),
            sizeof(MinidumpAnnotationList) + 3 * sizeof(MinidumpAnnotation) +
                3 * sizeof(MinidumpUTF8String) + 3 * sizeof(MinidumpByteArray) +
                strlen(kNames[0]) + 1 + kValues[0].size() + 2 +
                strlen(kNames[1]) + 1 + 3 + kValues[1].size() + 1 +
                strlen(kNames[2]) + 1 + 3 + kValues[2].size() + 2);

  auto* list = MinidumpAnnotationListAtStart(string_file.string(), 3);
  ASSERT_TRUE(list);
  EXPECT_EQ(list->count, 3u);

  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(MinidumpUTF8StringAtRVAAsString(string_file.string(),
                                              list->objects[i].name),
              kNames[i]);
    EXPECT_EQ(list->objects[i].type, kTypes[i]);
    EXPECT_EQ(list->objects[i].reserved, 0u);
    EXPECT_EQ(

        MinidumpByteArrayAtRVA(string_file.string(), list->objects[i].value),
        kValues[i]);
  }
}

TEST(MinidumpAnnotationWriter, DuplicateNames) {
  StringFile string_file;

  const char kName[] = "@@name!";
  const uint16_t kType = 0x1;
  const std::vector<uint8_t> kValue1{'r', 'e', 'd', '\0'};
  const std::vector<uint8_t> kValue2{'m', 'a', 'g', 'e', 'n', 't', 'a', '\0'};

  MinidumpAnnotationListWriter list_writer;

  auto annotation = std::make_unique<MinidumpAnnotationWriter>();
  annotation->InitializeWithData(kName, kType, kValue1);
  list_writer.AddObject(std::move(annotation));

  annotation = std::make_unique<MinidumpAnnotationWriter>();
  annotation->InitializeWithData(kName, kType, kValue2);
  list_writer.AddObject(std::move(annotation));

  EXPECT_TRUE(list_writer.WriteEverything(&string_file));

  ASSERT_EQ(string_file.string().size(),
            sizeof(MinidumpAnnotationList) + 2 * sizeof(MinidumpAnnotation) +
                2 * sizeof(MinidumpUTF8String) + 2 * sizeof(MinidumpByteArray) +
                2 * sizeof(kName) + kValue1.size() + kValue2.size());

  auto* list = MinidumpAnnotationListAtStart(string_file.string(), 2);
  ASSERT_TRUE(list);
  EXPECT_EQ(list->count, 2u);

  EXPECT_EQ(MinidumpUTF8StringAtRVAAsString(string_file.string(),
                                            list->objects[0].name),
            kName);
  EXPECT_EQ(list->objects[0].type, kType);
  EXPECT_EQ(

      MinidumpByteArrayAtRVA(string_file.string(), list->objects[0].value),
      kValue1);

  EXPECT_EQ(MinidumpUTF8StringAtRVAAsString(string_file.string(),
                                            list->objects[1].name),
            kName);
  EXPECT_EQ(list->objects[1].type, kType);
  EXPECT_EQ(

      MinidumpByteArrayAtRVA(string_file.string(), list->objects[1].value),
      kValue2);
}

}  // namespace
}  // namespace test
}  // namespace crashpad
