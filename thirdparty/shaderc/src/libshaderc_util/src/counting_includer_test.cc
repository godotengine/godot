// Copyright 2015 The Shaderc Authors. All rights reserved.
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

#include "libshaderc_util/counting_includer.h"

#include <thread>
#include <vector>

#include <gmock/gmock.h>

namespace {

// A trivial implementation of CountingIncluder's virtual methods, so tests can
// instantiate.
class ConcreteCountingIncluder : public shaderc_util::CountingIncluder {
 public:
  using IncludeResult = glslang::TShader::Includer::IncludeResult;
  ~ConcreteCountingIncluder() {
    // Avoid leaks.
    for (auto result : results_) {
      release_delegate(result);
    }
  }
  virtual IncludeResult* include_delegate(
      const char* requested, const char* requestor, IncludeType,
      size_t) override {
    const char kError[] = "Unexpected #include";
    results_.push_back(new IncludeResult{"", kError, strlen(kError), nullptr});
    return results_.back();
  }
  virtual void release_delegate(IncludeResult* include_result) override {
    delete include_result;
  }

 private:
  // All the results we've returned so far.
  std::vector<IncludeResult*> results_;
};

TEST(CountingIncluderTest, InitialCount) {
  EXPECT_EQ(0, ConcreteCountingIncluder().num_include_directives());
}

TEST(CountingIncluderTest, OneIncludeLocal) {
  ConcreteCountingIncluder includer;
  includer.includeLocal("random file name", "from me", 0);
  EXPECT_EQ(1, includer.num_include_directives());
}

TEST(CountingIncluderTest, TwoIncludesAnyIncludeType) {
  ConcreteCountingIncluder includer;
  includer.includeSystem("name1", "from me", 0);
  includer.includeLocal("name2", "me", 0);
  EXPECT_EQ(2, includer.num_include_directives());
}

TEST(CountingIncluderTest, ManyIncludes) {
  ConcreteCountingIncluder includer;
  for (int i = 0; i < 100; ++i) {
    includer.includeLocal("filename", "from me", i);
    includer.includeSystem("filename", "from me", i);
  }
  EXPECT_EQ(200, includer.num_include_directives());
}

#ifndef SHADERC_DISABLE_THREADED_TESTS
TEST(CountingIncluderTest, ThreadedIncludes) {
  ConcreteCountingIncluder includer;
  std::thread t1(
      [&includer]() { includer.includeLocal("name1", "me", 0); });
  std::thread t2(
      [&includer]() { includer.includeSystem("name2", "me", 1); });
  std::thread t3(
      [&includer]() { includer.includeLocal("name3", "me", 2); });
  t1.join();
  t2.join();
  t3.join();
  EXPECT_EQ(3, includer.num_include_directives());
}
#endif // SHADERC_DISABLE_THREADED_TESTS

}  // anonymous namespace
