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

#include <string>
#include <vector>

#include "base/rand_util.h"
#include "client/crashpad_info.h"
#include "gtest/gtest.h"
#include "util/misc/clock.h"
#include "util/thread/thread.h"

namespace crashpad {
namespace test {
namespace {

TEST(AnnotationListStatic, Register) {
  ASSERT_FALSE(AnnotationList::Get());
  EXPECT_TRUE(AnnotationList::Register());
  EXPECT_TRUE(AnnotationList::Get());
  EXPECT_EQ(AnnotationList::Get(), AnnotationList::Register());

  // This isn't expected usage of the AnnotationList API, but it is necessary
  // for testing.
  AnnotationList* list = AnnotationList::Get();
  CrashpadInfo::GetCrashpadInfo()->set_annotations_list(nullptr);
  delete list;

  EXPECT_FALSE(AnnotationList::Get());
}

class AnnotationList : public testing::Test {
 public:
  void SetUp() override {
    CrashpadInfo::GetCrashpadInfo()->set_annotations_list(&annotations_);
  }

  void TearDown() override {
    CrashpadInfo::GetCrashpadInfo()->set_annotations_list(nullptr);
  }

  // NOTE: Annotations should be declared at file-scope, but in order to test
  // them, they are declared as part of the test. These members are public so
  // they are accessible from global helpers.
  crashpad::StringAnnotation<8> one_{"First"};
  crashpad::StringAnnotation<256> two_{"Second"};
  crashpad::StringAnnotation<101> three_{"First"};

 protected:
  using AllAnnotations = std::vector<std::pair<std::string, std::string>>;

  AllAnnotations CollectAnnotations() {
    AllAnnotations annotations;

    for (Annotation* curr : annotations_) {
      if (!curr->is_set())
        continue;
      std::string value(static_cast<const char*>(curr->value()), curr->size());
      annotations.push_back(std::make_pair(curr->name(), value));
    }

    return annotations;
  }

  bool ContainsNameValue(const AllAnnotations& annotations,
                         const std::string& name,
                         const std::string& value) {
    return std::find(annotations.begin(),
                     annotations.end(),
                     std::make_pair(name, value)) != annotations.end();
  }

  crashpad::AnnotationList annotations_;
};

TEST_F(AnnotationList, SetAndClear) {
  one_.Set("this is a value longer than 8 bytes");
  AllAnnotations annotations = CollectAnnotations();

  EXPECT_EQ(1u, annotations.size());
  EXPECT_TRUE(ContainsNameValue(annotations, "First", "this is "));

  one_.Clear();

  EXPECT_EQ(0u, CollectAnnotations().size());

  one_.Set("short");
  two_.Set(std::string(500, 'A').data());

  annotations = CollectAnnotations();
  EXPECT_EQ(2u, annotations.size());

  EXPECT_EQ(5u, one_.size());
  EXPECT_EQ(256u, two_.size());

  EXPECT_TRUE(ContainsNameValue(annotations, "First", "short"));
  EXPECT_TRUE(ContainsNameValue(annotations, "Second", std::string(256, 'A')));
}

TEST_F(AnnotationList, DuplicateKeys) {
  ASSERT_EQ(0u, CollectAnnotations().size());

  one_.Set("1");
  three_.Set("2");

  AllAnnotations annotations = CollectAnnotations();
  EXPECT_EQ(2u, annotations.size());

  EXPECT_TRUE(ContainsNameValue(annotations, "First", "1"));
  EXPECT_TRUE(ContainsNameValue(annotations, "First", "2"));

  one_.Clear();

  annotations = CollectAnnotations();
  EXPECT_EQ(1u, annotations.size());
}

class RaceThread : public Thread {
 public:
  explicit RaceThread(test::AnnotationList* test) : Thread(), test_(test) {}

 private:
  void ThreadMain() override {
    for (int i = 0; i <= 50; ++i) {
      if (i % 2 == 0) {
        test_->three_.Set("three");
        test_->two_.Clear();
      } else {
        test_->three_.Clear();
      }
      SleepNanoseconds(base::RandInt(1, 1000));
    }
  }

  test::AnnotationList* test_;
};

TEST_F(AnnotationList, MultipleThreads) {
  ASSERT_EQ(0u, CollectAnnotations().size());

  RaceThread other_thread(this);
  other_thread.Start();

  for (int i = 0; i <= 50; ++i) {
    if (i % 2 == 0) {
      one_.Set("one");
      two_.Set("two");
    } else {
      one_.Clear();
    }
    SleepNanoseconds(base::RandInt(1, 1000));
  }

  other_thread.Join();

  AllAnnotations annotations = CollectAnnotations();
  EXPECT_GE(annotations.size(), 2u);
  EXPECT_LE(annotations.size(), 3u);

  EXPECT_TRUE(ContainsNameValue(annotations, "First", "one"));
  EXPECT_TRUE(ContainsNameValue(annotations, "First", "three"));

  if (annotations.size() == 3) {
    EXPECT_TRUE(ContainsNameValue(annotations, "Second", "two"));
  }
}

}  // namespace
}  // namespace test
}  // namespace crashpad
