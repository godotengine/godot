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

#include "util/misc/random_string.h"

#include <sys/types.h>

#include <set>

#include "base/macros.h"
#include "gtest/gtest.h"

namespace crashpad {
namespace test {
namespace {

TEST(RandomString, RandomString) {
  // Explicitly list the allowed characters, rather than relying on a range.
  // This prevents the test from having any dependency on the character set, so
  // that the implementation is free to assume all uppercase letters are
  // contiguous as in ASCII.
  const std::string allowed_characters("ABCDEFGHIJKLMNOPQRSTUVWXYZ");

  size_t character_counts[26] = {};
  ASSERT_EQ(allowed_characters.size(), arraysize(character_counts));

  std::set<std::string> strings;

  for (size_t i = 0; i < 256; ++i) {
    const std::string random_string = RandomString();
    EXPECT_EQ(random_string.size(), 16u);

    // Make sure that the string is unique. It is possible, but extremely
    // unlikely, for there to be collisions.
    auto result = strings.insert(random_string);
    EXPECT_TRUE(result.second) << random_string;

    for (char c : random_string) {
      size_t character_index = allowed_characters.find(c);

      // Make sure that no unexpected characters appear.
      EXPECT_NE(character_index, std::string::npos) << c;

      if (character_index != std::string::npos) {
        ++character_counts[character_index];
      }
    }
  }

  // Make sure every character appears at least once. It is possible, but
  // extremely unlikely, for a character to not appear at all.
  for (size_t character_index = 0;
       character_index < arraysize(character_counts);
       ++character_index) {
    EXPECT_GT(character_counts[character_index], 0u)
        << allowed_characters[character_index];
  }
}

}  // namespace
}  // namespace test
}  // namespace crashpad
