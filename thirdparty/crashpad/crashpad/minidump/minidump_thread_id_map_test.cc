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

#include "minidump/minidump_thread_id_map.h"

#include <sys/types.h>

#include <vector>

#include "base/macros.h"
#include "gtest/gtest.h"
#include "snapshot/test/test_thread_snapshot.h"

namespace crashpad {
namespace test {
namespace {

class MinidumpThreadIDMapTest : public testing::Test {
 public:
  MinidumpThreadIDMapTest()
      : Test(),
        thread_snapshots_(),
        test_thread_snapshots_() {
  }

  ~MinidumpThreadIDMapTest() override {}

  // testing::Test:
  void SetUp() override {
    for (size_t index = 0; index < arraysize(test_thread_snapshots_); ++index) {
      thread_snapshots_.push_back(&test_thread_snapshots_[index]);
    }
  }

 protected:
  static bool MapHasKeyValue(
      const MinidumpThreadIDMap* map, uint64_t key, uint32_t expected_value) {
    auto iterator = map->find(key);
    if (iterator == map->end()) {
      EXPECT_NE(iterator, map->end());
      return false;
    }
    if (iterator->second != expected_value) {
      EXPECT_EQ(iterator->second, expected_value);
      return false;
    }
    return true;
  }

  void SetThreadID(size_t index, uint64_t thread_id) {
    ASSERT_LT(index, arraysize(test_thread_snapshots_));
    test_thread_snapshots_[index].SetThreadID(thread_id);
  }

  const std::vector<const ThreadSnapshot*>& thread_snapshots() const {
    return thread_snapshots_;
  }

 private:
  std::vector<const ThreadSnapshot*> thread_snapshots_;
  TestThreadSnapshot test_thread_snapshots_[5];

  DISALLOW_COPY_AND_ASSIGN(MinidumpThreadIDMapTest);
};

TEST_F(MinidumpThreadIDMapTest, NoThreads) {
  // Don’t use thread_snapshots(), because it’s got some threads in it, and the
  // point of this test is to make sure that BuildMinidumpThreadIDMap() works
  // with no threads.
  std::vector<const ThreadSnapshot*> thread_snapshots;
  MinidumpThreadIDMap thread_id_map;
  BuildMinidumpThreadIDMap(thread_snapshots, &thread_id_map);

  EXPECT_TRUE(thread_id_map.empty());
}

TEST_F(MinidumpThreadIDMapTest, SimpleMapping) {
  SetThreadID(0, 1);
  SetThreadID(1, 3);
  SetThreadID(2, 5);
  SetThreadID(3, 7);
  SetThreadID(4, 9);

  MinidumpThreadIDMap thread_id_map;
  BuildMinidumpThreadIDMap(thread_snapshots(), &thread_id_map);

  EXPECT_EQ(thread_id_map.size(), 5u);
  EXPECT_PRED3(MapHasKeyValue, &thread_id_map, 1, 1);
  EXPECT_PRED3(MapHasKeyValue, &thread_id_map, 3, 3);
  EXPECT_PRED3(MapHasKeyValue, &thread_id_map, 5, 5);
  EXPECT_PRED3(MapHasKeyValue, &thread_id_map, 7, 7);
  EXPECT_PRED3(MapHasKeyValue, &thread_id_map, 9, 9);
}

TEST_F(MinidumpThreadIDMapTest, Truncation) {
  SetThreadID(0, 0x0000000000000000);
  SetThreadID(1, 0x9999999900000001);
  SetThreadID(2, 0x9999999980000001);
  SetThreadID(3, 0x99999999fffffffe);
  SetThreadID(4, 0x99999999ffffffff);

  MinidumpThreadIDMap thread_id_map;
  BuildMinidumpThreadIDMap(thread_snapshots(), &thread_id_map);

  EXPECT_EQ(thread_id_map.size(), 5u);
  EXPECT_PRED3(MapHasKeyValue, &thread_id_map, 0x0000000000000000, 0x00000000);
  EXPECT_PRED3(MapHasKeyValue, &thread_id_map, 0x9999999900000001, 0x00000001);
  EXPECT_PRED3(MapHasKeyValue, &thread_id_map, 0x9999999980000001, 0x80000001);
  EXPECT_PRED3(MapHasKeyValue, &thread_id_map, 0x99999999fffffffe, 0xfffffffe);
  EXPECT_PRED3(MapHasKeyValue, &thread_id_map, 0x99999999ffffffff, 0xffffffff);
}

TEST_F(MinidumpThreadIDMapTest, DuplicateThreadID) {
  SetThreadID(0, 2);
  SetThreadID(1, 4);
  SetThreadID(2, 4);
  SetThreadID(3, 6);
  SetThreadID(4, 8);

  MinidumpThreadIDMap thread_id_map;
  BuildMinidumpThreadIDMap(thread_snapshots(), &thread_id_map);

  EXPECT_EQ(thread_id_map.size(), 4u);
  EXPECT_PRED3(MapHasKeyValue, &thread_id_map, 2, 2);
  EXPECT_PRED3(MapHasKeyValue, &thread_id_map, 4, 4);
  EXPECT_PRED3(MapHasKeyValue, &thread_id_map, 6, 6);
  EXPECT_PRED3(MapHasKeyValue, &thread_id_map, 8, 8);
}

TEST_F(MinidumpThreadIDMapTest, Collision) {
  SetThreadID(0, 0x0000000000000010);
  SetThreadID(1, 0x0000000000000020);
  SetThreadID(2, 0x0000000000000030);
  SetThreadID(3, 0x0000000000000040);
  SetThreadID(4, 0x0000000100000010);

  MinidumpThreadIDMap thread_id_map;
  BuildMinidumpThreadIDMap(thread_snapshots(), &thread_id_map);

  EXPECT_EQ(thread_id_map.size(), 5u);
  EXPECT_PRED3(MapHasKeyValue, &thread_id_map, 0x0000000000000010, 0);
  EXPECT_PRED3(MapHasKeyValue, &thread_id_map, 0x0000000000000020, 1);
  EXPECT_PRED3(MapHasKeyValue, &thread_id_map, 0x0000000000000030, 2);
  EXPECT_PRED3(MapHasKeyValue, &thread_id_map, 0x0000000000000040, 3);
  EXPECT_PRED3(MapHasKeyValue, &thread_id_map, 0x0000000100000010, 4);
}

TEST_F(MinidumpThreadIDMapTest, DuplicateAndCollision) {
  SetThreadID(0, 0x0000000100000010);
  SetThreadID(1, 0x0000000000000010);
  SetThreadID(2, 0x0000000000000020);
  SetThreadID(3, 0x0000000000000030);
  SetThreadID(4, 0x0000000000000020);

  MinidumpThreadIDMap thread_id_map;
  BuildMinidumpThreadIDMap(thread_snapshots(), &thread_id_map);

  EXPECT_EQ(thread_id_map.size(), 4u);
  EXPECT_PRED3(MapHasKeyValue, &thread_id_map, 0x0000000100000010, 0);
  EXPECT_PRED3(MapHasKeyValue, &thread_id_map, 0x0000000000000010, 1);
  EXPECT_PRED3(MapHasKeyValue, &thread_id_map, 0x0000000000000020, 2);
  EXPECT_PRED3(MapHasKeyValue, &thread_id_map, 0x0000000000000030, 3);
}

TEST_F(MinidumpThreadIDMapTest, AllDuplicates) {
  SetThreadID(0, 6);
  SetThreadID(1, 6);
  SetThreadID(2, 6);
  SetThreadID(3, 6);
  SetThreadID(4, 6);

  MinidumpThreadIDMap thread_id_map;
  BuildMinidumpThreadIDMap(thread_snapshots(), &thread_id_map);

  EXPECT_EQ(thread_id_map.size(), 1u);
  EXPECT_PRED3(MapHasKeyValue, &thread_id_map, 6, 6);
}

}  // namespace
}  // namespace test
}  // namespace crashpad
