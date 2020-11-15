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

#include "client/annotation.h"

#include <array>
#include <string>

#include "client/annotation_list.h"
#include "client/crashpad_info.h"
#include "gtest/gtest.h"
#include "test/gtest_death.h"

namespace crashpad {
namespace test {
namespace {

class Annotation : public testing::Test {
 public:
  void SetUp() override {
    CrashpadInfo::GetCrashpadInfo()->set_annotations_list(&annotations_);
  }

  void TearDown() override {
    CrashpadInfo::GetCrashpadInfo()->set_annotations_list(nullptr);
  }

  size_t AnnotationsCount() {
    size_t result = 0;
    for (auto* annotation : annotations_) {
      if (annotation->is_set())
        ++result;
    }
    return result;
  }

 protected:
  crashpad::AnnotationList annotations_;
};

TEST_F(Annotation, Basics) {
  constexpr crashpad::Annotation::Type kType =
      crashpad::Annotation::UserDefinedType(1);

  const char kName[] = "annotation 1";
  char buffer[1024];
  crashpad::Annotation annotation(kType, kName, buffer);

  EXPECT_FALSE(annotation.is_set());
  EXPECT_EQ(0u, AnnotationsCount());

  EXPECT_EQ(kType, annotation.type());
  EXPECT_EQ(0u, annotation.size());
  EXPECT_EQ(std::string(kName), annotation.name());
  EXPECT_EQ(buffer, annotation.value());

  annotation.SetSize(10);

  EXPECT_TRUE(annotation.is_set());
  EXPECT_EQ(1u, AnnotationsCount());

  EXPECT_EQ(10u, annotation.size());
  EXPECT_EQ(&annotation, *annotations_.begin());

  annotation.Clear();

  EXPECT_FALSE(annotation.is_set());
  EXPECT_EQ(0u, AnnotationsCount());

  EXPECT_EQ(0u, annotation.size());
}

TEST_F(Annotation, StringType) {
  crashpad::StringAnnotation<5> annotation("name");

  EXPECT_FALSE(annotation.is_set());

  EXPECT_EQ(crashpad::Annotation::Type::kString, annotation.type());
  EXPECT_EQ(0u, annotation.size());
  EXPECT_EQ(std::string("name"), annotation.name());
  EXPECT_EQ(0u, annotation.value().size());

  annotation.Set("test");

  EXPECT_TRUE(annotation.is_set());
  EXPECT_EQ(1u, AnnotationsCount());

  EXPECT_EQ(4u, annotation.size());
  EXPECT_EQ("test", annotation.value());

  annotation.Set(std::string("loooooooooooong"));

  EXPECT_TRUE(annotation.is_set());
  EXPECT_EQ(1u, AnnotationsCount());

  EXPECT_EQ(5u, annotation.size());
  EXPECT_EQ("loooo", annotation.value());
}

TEST(StringAnnotation, ArrayOfString) {
  static crashpad::StringAnnotation<4> annotations[] = {
      {"test-1", crashpad::StringAnnotation<4>::Tag::kArray},
      {"test-2", crashpad::StringAnnotation<4>::Tag::kArray},
      {"test-3", crashpad::StringAnnotation<4>::Tag::kArray},
      {"test-4", crashpad::StringAnnotation<4>::Tag::kArray},
  };

  for (auto& annotation : annotations) {
    EXPECT_FALSE(annotation.is_set());
  }
}

#if DCHECK_IS_ON()

TEST(AnnotationDeathTest, EmbeddedNUL) {
  crashpad::StringAnnotation<5> annotation("name");
  EXPECT_DEATH_CHECK(annotation.Set(std::string("te\0st", 5)), "embedded NUL");
}

#endif

}  // namespace
}  // namespace test
}  // namespace crashpad
