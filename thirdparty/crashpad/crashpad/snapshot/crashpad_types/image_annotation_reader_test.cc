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

#include "snapshot/crashpad_types/image_annotation_reader.h"

#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>

#include "base/logging.h"
#include "build/build_config.h"
#include "client/annotation.h"
#include "client/annotation_list.h"
#include "client/simple_string_dictionary.h"
#include "gtest/gtest.h"
#include "test/multiprocess_exec.h"
#include "test/process_type.h"
#include "util/file/file_io.h"
#include "util/misc/as_underlying_type.h"
#include "util/misc/from_pointer_cast.h"
#include "util/process/process_memory_native.h"

namespace crashpad {
namespace test {
namespace {

void ExpectSimpleMap(const std::map<std::string, std::string>& map,
                     const SimpleStringDictionary& expected_map) {
  EXPECT_EQ(map.size(), expected_map.GetCount());
  for (const auto& pair : map) {
    EXPECT_EQ(pair.second, expected_map.GetValueForKey(pair.first));
  }
}

void ExpectAnnotationList(const std::vector<AnnotationSnapshot>& list,
                          AnnotationList& expected_list) {
  size_t index = 0;
  for (const Annotation* expected_annotation : expected_list) {
    const AnnotationSnapshot& annotation = list[index++];
    EXPECT_EQ(annotation.name, expected_annotation->name());
    EXPECT_EQ(annotation.type, AsUnderlyingType(expected_annotation->type()));
    EXPECT_EQ(annotation.value.size(), expected_annotation->size());
    EXPECT_EQ(memcmp(annotation.value.data(),
                     expected_annotation->value(),
                     std::min(VMSize{annotation.value.size()},
                              VMSize{expected_annotation->size()})),
              0);
  }
}

void BuildTestStructures(
    std::vector<std::unique_ptr<Annotation>>* annotations_storage,
    SimpleStringDictionary* into_map,
    AnnotationList* into_annotation_list) {
  into_map->SetKeyValue("key", "value");
  into_map->SetKeyValue("key2", "value2");

  static constexpr char kAnnotationName[] = "test annotation";
  static constexpr char kAnnotationValue[] = "test annotation value";
  annotations_storage->push_back(std::make_unique<Annotation>(
      Annotation::Type::kString,
      kAnnotationName,
      reinterpret_cast<void*>(const_cast<char*>(kAnnotationValue))));
  annotations_storage->back()->SetSize(sizeof(kAnnotationValue));
  into_annotation_list->Add(annotations_storage->back().get());

  static constexpr char kAnnotationName2[] = "test annotation2";
  static constexpr char kAnnotationValue2[] = "test annotation value2";
  annotations_storage->push_back(std::make_unique<Annotation>(
      Annotation::Type::kString,
      kAnnotationName2,
      reinterpret_cast<void*>(const_cast<char*>(kAnnotationValue2))));
  annotations_storage->back()->SetSize(sizeof(kAnnotationValue2));
  into_annotation_list->Add(annotations_storage->back().get());
}

void ExpectAnnotations(ProcessType process,
                       bool is_64_bit,
                       VMAddress simple_map_address,
                       VMAddress annotation_list_address) {
  ProcessMemoryNative memory;
  ASSERT_TRUE(memory.Initialize(process));

  ProcessMemoryRange range;
  ASSERT_TRUE(range.Initialize(&memory, is_64_bit));

  SimpleStringDictionary expected_simple_map;
  std::vector<std::unique_ptr<Annotation>> storage;
  AnnotationList expected_annotations;
  BuildTestStructures(&storage, &expected_simple_map, &expected_annotations);

  ImageAnnotationReader reader(&range);

  std::map<std::string, std::string> simple_map;
  ASSERT_TRUE(reader.SimpleMap(simple_map_address, &simple_map));
  ExpectSimpleMap(simple_map, expected_simple_map);

  std::vector<AnnotationSnapshot> annotation_list;
  ASSERT_TRUE(
      reader.AnnotationsList(annotation_list_address, &annotation_list));
  ExpectAnnotationList(annotation_list, expected_annotations);
}

TEST(ImageAnnotationReader, ReadFromSelf) {
  SimpleStringDictionary map;
  std::vector<std::unique_ptr<Annotation>> storage;
  AnnotationList annotations;
  BuildTestStructures(&storage, &map, &annotations);

#if defined(ARCH_CPU_64_BITS)
  constexpr bool am_64_bit = true;
#else
  constexpr bool am_64_bit = false;
#endif

  ExpectAnnotations(GetSelfProcess(),
                    am_64_bit,
                    FromPointerCast<VMAddress>(&map),
                    FromPointerCast<VMAddress>(&annotations));
}

CRASHPAD_CHILD_TEST_MAIN(ReadAnnotationsFromChildTestMain) {
  SimpleStringDictionary map;
  std::vector<std::unique_ptr<Annotation>> storage;
  AnnotationList annotations;
  BuildTestStructures(&storage, &map, &annotations);

  VMAddress simple_map_address = FromPointerCast<VMAddress>(&map);
  VMAddress annotations_address = FromPointerCast<VMAddress>(&annotations);
  FileHandle out = StdioFileHandle(StdioStream::kStandardOutput);
  CheckedWriteFile(out, &simple_map_address, sizeof(simple_map_address));
  CheckedWriteFile(out, &annotations_address, sizeof(annotations_address));

  CheckedReadFileAtEOF(StdioFileHandle(StdioStream::kStandardInput));
  return 0;
}

class ReadFromChildTest : public MultiprocessExec {
 public:
  ReadFromChildTest() : MultiprocessExec() {
    SetChildTestMainFunction("ReadAnnotationsFromChildTestMain");
  }

  ~ReadFromChildTest() = default;

 private:
  void MultiprocessParent() {
#if defined(ARCH_CPU_64_BITS)
    constexpr bool am_64_bit = true;
#else
    constexpr bool am_64_bit = false;
#endif

    VMAddress simple_map_address;
    VMAddress annotations_address;
    ASSERT_TRUE(ReadFileExactly(
        ReadPipeHandle(), &simple_map_address, sizeof(simple_map_address)));
    ASSERT_TRUE(ReadFileExactly(
        ReadPipeHandle(), &annotations_address, sizeof(annotations_address)));
    ExpectAnnotations(
        ChildProcess(), am_64_bit, simple_map_address, annotations_address);
  }

  DISALLOW_COPY_AND_ASSIGN(ReadFromChildTest);
};

TEST(ImageAnnotationReader, ReadFromChild) {
  ReadFromChildTest test;
  test.Run();
}

}  // namespace
}  // namespace test
}  // namespace crashpad
