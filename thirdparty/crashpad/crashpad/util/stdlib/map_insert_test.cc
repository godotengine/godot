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

#include "util/stdlib/map_insert.h"

#include <string>

#include "gtest/gtest.h"

namespace crashpad {
namespace test {
namespace {

TEST(MapInsert, MapInsertOrReplace) {
  std::map<std::string, int> map;
  int old_value;
  EXPECT_TRUE(MapInsertOrReplace(&map, "key", 1, &old_value));
  std::map<std::string, int> expect_map;
  expect_map["key"] = 1;
  EXPECT_EQ(map, expect_map);

  EXPECT_FALSE(MapInsertOrReplace(&map, "key", 2, &old_value));
  EXPECT_EQ(old_value, 1);
  expect_map["key"] = 2;
  EXPECT_EQ(map, expect_map);

  EXPECT_TRUE(MapInsertOrReplace(&map, "another", 3, &old_value));
  expect_map["another"] = 3;
  EXPECT_EQ(map, expect_map);

  // Make sure nullptr is accepted as old_value.
  EXPECT_TRUE(MapInsertOrReplace(&map, "yet another", 5, nullptr));
  expect_map["yet another"] = 5;
  EXPECT_EQ(map, expect_map);

  EXPECT_FALSE(MapInsertOrReplace(&map, "yet another", 6, nullptr));
  expect_map["yet another"] = 6;
  EXPECT_EQ(map, expect_map);
}

}  // namespace
}  // namespace test
}  // namespace crashpad
