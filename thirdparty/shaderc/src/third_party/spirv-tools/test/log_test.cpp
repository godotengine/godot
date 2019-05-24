// Copyright (c) 2016 Google Inc.
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

#include "source/opt/log.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace spvtools {
namespace {

using ::testing::MatchesRegex;

TEST(Log, Unimplemented) {
  int invocation = 0;
  auto consumer = [&invocation](spv_message_level_t level, const char* source,
                                const spv_position_t&, const char* message) {
    ++invocation;
    EXPECT_EQ(SPV_MSG_INTERNAL_ERROR, level);
    EXPECT_THAT(source, MatchesRegex(".*log_test.cpp$"));
    EXPECT_STREQ("unimplemented: the-ultimite-feature", message);
  };

  SPIRV_UNIMPLEMENTED(consumer, "the-ultimite-feature");
  EXPECT_EQ(1, invocation);
}

TEST(Log, Unreachable) {
  int invocation = 0;
  auto consumer = [&invocation](spv_message_level_t level, const char* source,
                                const spv_position_t&, const char* message) {
    ++invocation;
    EXPECT_EQ(SPV_MSG_INTERNAL_ERROR, level);
    EXPECT_THAT(source, MatchesRegex(".*log_test.cpp$"));
    EXPECT_STREQ("unreachable", message);
  };

  SPIRV_UNREACHABLE(consumer);
  EXPECT_EQ(1, invocation);
}

}  // namespace
}  // namespace spvtools
