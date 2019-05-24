// Copyright (c) 2017 Google Inc.
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

#include <algorithm>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "source/comp/move_to_front.h"

namespace spvtools {
namespace comp {
namespace {

// Class used to test the inner workings of MoveToFront.
class MoveToFrontTester : public MoveToFront {
 public:
  // Inserts the value in the internal tree data structure. For testing only.
  void TestInsert(uint32_t val) { InsertNode(CreateNode(val, val)); }

  // Removes the value from the internal tree data structure. For testing only.
  void TestRemove(uint32_t val) {
    const auto it = value_to_node_.find(val);
    assert(it != value_to_node_.end());
    RemoveNode(it->second);
  }

  // Prints the internal tree data structure to |out|. For testing only.
  void PrintTree(std::ostream& out, bool print_timestamp = false) const {
    if (root_) PrintTreeInternal(out, root_, 1, print_timestamp);
  }

  // Returns node handle corresponding to the value. The value may not be in the
  // tree.
  uint32_t GetNodeHandle(uint32_t value) const {
    const auto it = value_to_node_.find(value);
    if (it == value_to_node_.end()) return 0;

    return it->second;
  }

  // Returns total node count (both those in the tree and removed,
  // but not the NIL singleton).
  size_t GetTotalNodeCount() const {
    assert(nodes_.size());
    return nodes_.size() - 1;
  }

  uint32_t GetLastAccessedValue() const { return last_accessed_value_; }

 private:
  // Prints the internal tree data structure for debug purposes in the following
  // format:
  // 10H3S4----5H1S1-----D2
  //           15H2S2----12H1S1----D3
  // Right links are horizontal, left links step down one line.
  // 5H1S1 is read as value 5, height 1, size 1. Optionally node label can also
  // contain timestamp (5H1S1T15). D3 stands for depth 3.
  void PrintTreeInternal(std::ostream& out, uint32_t node, size_t depth,
                         bool print_timestamp) const;
};

void MoveToFrontTester::PrintTreeInternal(std::ostream& out, uint32_t node,
                                          size_t depth,
                                          bool print_timestamp) const {
  if (!node) {
    out << "D" << depth - 1 << std::endl;
    return;
  }

  const size_t kTextFieldWvaluethWithoutTimestamp = 10;
  const size_t kTextFieldWvaluethWithTimestamp = 14;
  const size_t text_field_wvalueth = print_timestamp
                                         ? kTextFieldWvaluethWithTimestamp
                                         : kTextFieldWvaluethWithoutTimestamp;

  std::stringstream label;
  label << ValueOf(node) << "H" << HeightOf(node) << "S" << SizeOf(node);
  if (print_timestamp) label << "T" << TimestampOf(node);
  const size_t label_length = label.str().length();
  if (label_length < text_field_wvalueth)
    label << std::string(text_field_wvalueth - label_length, '-');

  out << label.str();

  PrintTreeInternal(out, RightOf(node), depth + 1, print_timestamp);

  if (LeftOf(node)) {
    out << std::string(depth * text_field_wvalueth, ' ');
    PrintTreeInternal(out, LeftOf(node), depth + 1, print_timestamp);
  }
}

void CheckTree(const MoveToFrontTester& mtf, const std::string& expected,
               bool print_timestamp = false) {
  std::stringstream ss;
  mtf.PrintTree(ss, print_timestamp);
  EXPECT_EQ(expected, ss.str());
}

TEST(MoveToFront, EmptyTree) {
  MoveToFrontTester mtf;
  CheckTree(mtf, std::string());
}

TEST(MoveToFront, InsertLeftRotation) {
  MoveToFrontTester mtf;

  mtf.TestInsert(30);
  mtf.TestInsert(20);

  CheckTree(mtf, std::string(R"(
30H2S2----20H1S1----D2
)")
                     .substr(1));

  mtf.TestInsert(10);
  CheckTree(mtf, std::string(R"(
20H2S3----10H1S1----D2
          30H1S1----D2
)")
                     .substr(1));
}

TEST(MoveToFront, InsertRightRotation) {
  MoveToFrontTester mtf;

  mtf.TestInsert(10);
  mtf.TestInsert(20);

  CheckTree(mtf, std::string(R"(
10H2S2----D1
          20H1S1----D2
)")
                     .substr(1));

  mtf.TestInsert(30);
  CheckTree(mtf, std::string(R"(
20H2S3----10H1S1----D2
          30H1S1----D2
)")
                     .substr(1));
}

TEST(MoveToFront, InsertRightLeftRotation) {
  MoveToFrontTester mtf;

  mtf.TestInsert(30);
  mtf.TestInsert(20);

  CheckTree(mtf, std::string(R"(
30H2S2----20H1S1----D2
)")
                     .substr(1));

  mtf.TestInsert(25);
  CheckTree(mtf, std::string(R"(
25H2S3----20H1S1----D2
          30H1S1----D2
)")
                     .substr(1));
}

TEST(MoveToFront, InsertLeftRightRotation) {
  MoveToFrontTester mtf;

  mtf.TestInsert(10);
  mtf.TestInsert(20);

  CheckTree(mtf, std::string(R"(
10H2S2----D1
          20H1S1----D2
)")
                     .substr(1));

  mtf.TestInsert(15);
  CheckTree(mtf, std::string(R"(
15H2S3----10H1S1----D2
          20H1S1----D2
)")
                     .substr(1));
}

TEST(MoveToFront, RemoveSingleton) {
  MoveToFrontTester mtf;

  mtf.TestInsert(10);
  CheckTree(mtf, std::string(R"(
10H1S1----D1
)")
                     .substr(1));

  mtf.TestRemove(10);
  CheckTree(mtf, "");
}

TEST(MoveToFront, RemoveRootWithScapegoat) {
  MoveToFrontTester mtf;

  mtf.TestInsert(10);
  mtf.TestInsert(5);
  mtf.TestInsert(15);
  CheckTree(mtf, std::string(R"(
10H2S3----5H1S1-----D2
          15H1S1----D2
)")
                     .substr(1));

  mtf.TestRemove(10);
  CheckTree(mtf, std::string(R"(
15H2S2----5H1S1-----D2
)")
                     .substr(1));
}

TEST(MoveToFront, RemoveRightRotation) {
  MoveToFrontTester mtf;

  mtf.TestInsert(10);
  mtf.TestInsert(5);
  mtf.TestInsert(15);
  mtf.TestInsert(20);
  CheckTree(mtf, std::string(R"(
10H3S4----5H1S1-----D2
          15H2S2----D2
                    20H1S1----D3
)")
                     .substr(1));

  mtf.TestRemove(5);

  CheckTree(mtf, std::string(R"(
15H2S3----10H1S1----D2
          20H1S1----D2
)")
                     .substr(1));
}

TEST(MoveToFront, RemoveLeftRotation) {
  MoveToFrontTester mtf;

  mtf.TestInsert(10);
  mtf.TestInsert(15);
  mtf.TestInsert(5);
  mtf.TestInsert(1);
  CheckTree(mtf, std::string(R"(
10H3S4----5H2S2-----1H1S1-----D3
          15H1S1----D2
)")
                     .substr(1));

  mtf.TestRemove(15);

  CheckTree(mtf, std::string(R"(
5H2S3-----1H1S1-----D2
          10H1S1----D2
)")
                     .substr(1));
}

TEST(MoveToFront, RemoveLeftRightRotation) {
  MoveToFrontTester mtf;

  mtf.TestInsert(10);
  mtf.TestInsert(15);
  mtf.TestInsert(5);
  mtf.TestInsert(12);
  CheckTree(mtf, std::string(R"(
10H3S4----5H1S1-----D2
          15H2S2----12H1S1----D3
)")
                     .substr(1));

  mtf.TestRemove(5);

  CheckTree(mtf, std::string(R"(
12H2S3----10H1S1----D2
          15H1S1----D2
)")
                     .substr(1));
}

TEST(MoveToFront, RemoveRightLeftRotation) {
  MoveToFrontTester mtf;

  mtf.TestInsert(10);
  mtf.TestInsert(15);
  mtf.TestInsert(5);
  mtf.TestInsert(8);
  CheckTree(mtf, std::string(R"(
10H3S4----5H2S2-----D2
                    8H1S1-----D3
          15H1S1----D2
)")
                     .substr(1));

  mtf.TestRemove(15);

  CheckTree(mtf, std::string(R"(
8H2S3-----5H1S1-----D2
          10H1S1----D2
)")
                     .substr(1));
}

TEST(MoveToFront, MultipleOperations) {
  MoveToFrontTester mtf;
  std::vector<uint32_t> vals = {5, 11, 12, 16, 15, 6, 14, 2,
                                7, 10, 4,  8,  9,  3, 1,  13};

  for (uint32_t i : vals) {
    mtf.TestInsert(i);
  }

  CheckTree(mtf, std::string(R"(
11H5S16---5H4S10----3H3S4-----2H2S2-----1H1S1-----D5
                              4H1S1-----D4
                    7H3S5-----6H1S1-----D4
                              9H2S3-----8H1S1-----D5
                                        10H1S1----D5
          15H3S5----13H2S3----12H1S1----D4
                              14H1S1----D4
                    16H1S1----D3
)")
                     .substr(1));

  mtf.TestRemove(11);

  CheckTree(mtf, std::string(R"(
10H5S15---5H4S9-----3H3S4-----2H2S2-----1H1S1-----D5
                              4H1S1-----D4
                    7H3S4-----6H1S1-----D4
                              9H2S2-----8H1S1-----D5
          15H3S5----13H2S3----12H1S1----D4
                              14H1S1----D4
                    16H1S1----D3
)")
                     .substr(1));

  mtf.TestInsert(11);

  CheckTree(mtf, std::string(R"(
10H5S16---5H4S9-----3H3S4-----2H2S2-----1H1S1-----D5
                              4H1S1-----D4
                    7H3S4-----6H1S1-----D4
                              9H2S2-----8H1S1-----D5
          13H3S6----12H2S2----11H1S1----D4
                    15H2S3----14H1S1----D4
                              16H1S1----D4
)")
                     .substr(1));

  mtf.TestRemove(5);

  CheckTree(mtf, std::string(R"(
10H5S15---6H4S8-----3H3S4-----2H2S2-----1H1S1-----D5
                              4H1S1-----D4
                    8H2S3-----7H1S1-----D4
                              9H1S1-----D4
          13H3S6----12H2S2----11H1S1----D4
                    15H2S3----14H1S1----D4
                              16H1S1----D4
)")
                     .substr(1));

  mtf.TestInsert(5);

  CheckTree(mtf, std::string(R"(
10H5S16---6H4S9-----3H3S5-----2H2S2-----1H1S1-----D5
                              4H2S2-----D4
                                        5H1S1-----D5
                    8H2S3-----7H1S1-----D4
                              9H1S1-----D4
          13H3S6----12H2S2----11H1S1----D4
                    15H2S3----14H1S1----D4
                              16H1S1----D4
)")
                     .substr(1));

  mtf.TestRemove(2);
  mtf.TestRemove(1);
  mtf.TestRemove(4);
  mtf.TestRemove(3);
  mtf.TestRemove(6);
  mtf.TestRemove(5);
  mtf.TestRemove(7);
  mtf.TestRemove(9);

  CheckTree(mtf, std::string(R"(
13H4S8----10H3S4----8H1S1-----D3
                    12H2S2----11H1S1----D4
          15H2S3----14H1S1----D3
                    16H1S1----D3
)")
                     .substr(1));
}

TEST(MoveToFront, BiggerScaleTreeTest) {
  MoveToFrontTester mtf;
  std::set<uint32_t> all_vals;

  const uint32_t kMagic1 = 2654435761;
  const uint32_t kMagic2 = 10000;

  for (uint32_t i = 1; i < 1000; ++i) {
    const uint32_t val = (i * kMagic1) % kMagic2;
    if (!all_vals.count(val)) {
      mtf.TestInsert(val);
      all_vals.insert(val);
    }
  }

  for (uint32_t i = 1; i < 1000; ++i) {
    const uint32_t val = (i * kMagic1) % kMagic2;
    if (val % 2 == 0) {
      mtf.TestRemove(val);
      all_vals.erase(val);
    }
  }

  for (uint32_t i = 1000; i < 2000; ++i) {
    const uint32_t val = (i * kMagic1) % kMagic2;
    if (!all_vals.count(val)) {
      mtf.TestInsert(val);
      all_vals.insert(val);
    }
  }

  for (uint32_t i = 1; i < 2000; ++i) {
    const uint32_t val = (i * kMagic1) % kMagic2;
    if (val > 50) {
      mtf.TestRemove(val);
      all_vals.erase(val);
    }
  }

  EXPECT_EQ(all_vals, std::set<uint32_t>({2, 4, 11, 13, 24, 33, 35, 37, 46}));

  CheckTree(mtf, std::string(R"(
33H4S9----11H3S5----2H2S2-----D3
                              4H1S1-----D4
                    13H2S2----D3
                              24H1S1----D4
          37H2S3----35H1S1----D3
                    46H1S1----D3
)")
                     .substr(1));
}

TEST(MoveToFront, RankFromValue) {
  MoveToFrontTester mtf;

  uint32_t rank = 0;
  EXPECT_FALSE(mtf.RankFromValue(1, &rank));

  EXPECT_TRUE(mtf.Insert(1));
  EXPECT_TRUE(mtf.Insert(2));
  EXPECT_TRUE(mtf.Insert(3));
  EXPECT_FALSE(mtf.Insert(2));
  CheckTree(mtf,
            std::string(R"(
2H2S3T2-------1H1S1T1-------D2
              3H1S1T3-------D2
)")
                .substr(1),
            /* print_timestamp = */ true);

  EXPECT_FALSE(mtf.RankFromValue(4, &rank));

  EXPECT_TRUE(mtf.RankFromValue(1, &rank));
  EXPECT_EQ(3u, rank);

  CheckTree(mtf,
            std::string(R"(
3H2S3T3-------2H1S1T2-------D2
              1H1S1T4-------D2
)")
                .substr(1),
            /* print_timestamp = */ true);

  EXPECT_TRUE(mtf.RankFromValue(1, &rank));
  EXPECT_EQ(1u, rank);

  EXPECT_TRUE(mtf.RankFromValue(3, &rank));
  EXPECT_EQ(2u, rank);

  EXPECT_TRUE(mtf.RankFromValue(2, &rank));
  EXPECT_EQ(3u, rank);

  EXPECT_TRUE(mtf.Insert(40));

  EXPECT_TRUE(mtf.RankFromValue(1, &rank));
  EXPECT_EQ(4u, rank);

  EXPECT_TRUE(mtf.Insert(50));

  EXPECT_TRUE(mtf.RankFromValue(1, &rank));
  EXPECT_EQ(2u, rank);

  CheckTree(mtf,
            std::string(R"(
2H3S5T6-------3H1S1T5-------D2
              50H2S3T9------40H1S1T7------D3
                            1H1S1T10------D3
)")
                .substr(1),
            /* print_timestamp = */ true);

  EXPECT_TRUE(mtf.RankFromValue(50, &rank));
  EXPECT_EQ(2u, rank);

  EXPECT_EQ(5u, mtf.GetSize());
  CheckTree(mtf,
            std::string(R"(
2H3S5T6-------3H1S1T5-------D2
              1H2S3T10------40H1S1T7------D3
                            50H1S1T11-----D3
)")
                .substr(1),
            /* print_timestamp = */ true);

  EXPECT_FALSE(mtf.RankFromValue(0, &rank));
  EXPECT_FALSE(mtf.RankFromValue(20, &rank));
}

TEST(MoveToFront, ValueFromRank) {
  MoveToFrontTester mtf;

  uint32_t value = 0;
  EXPECT_FALSE(mtf.ValueFromRank(0, &value));
  EXPECT_FALSE(mtf.ValueFromRank(1, &value));

  EXPECT_TRUE(mtf.Insert(1));
  EXPECT_EQ(1u, mtf.GetLastAccessedValue());
  EXPECT_TRUE(mtf.Insert(2));
  EXPECT_EQ(2u, mtf.GetLastAccessedValue());
  EXPECT_TRUE(mtf.Insert(3));
  EXPECT_EQ(3u, mtf.GetLastAccessedValue());

  EXPECT_TRUE(mtf.ValueFromRank(3, &value));
  EXPECT_EQ(1u, value);
  EXPECT_EQ(1u, mtf.GetLastAccessedValue());

  EXPECT_TRUE(mtf.ValueFromRank(1, &value));
  EXPECT_EQ(1u, value);
  EXPECT_EQ(1u, mtf.GetLastAccessedValue());

  CheckTree(mtf,
            std::string(R"(
3H2S3T3-------2H1S1T2-------D2
              1H1S1T4-------D2
)")
                .substr(1),
            /* print_timestamp = */ true);

  EXPECT_TRUE(mtf.ValueFromRank(2, &value));
  EXPECT_EQ(3u, value);

  EXPECT_EQ(3u, mtf.GetSize());

  CheckTree(mtf,
            std::string(R"(
1H2S3T4-------2H1S1T2-------D2
              3H1S1T5-------D2
)")
                .substr(1),
            /* print_timestamp = */ true);

  EXPECT_TRUE(mtf.ValueFromRank(3, &value));
  EXPECT_EQ(2u, value);

  CheckTree(mtf,
            std::string(R"(
3H2S3T5-------1H1S1T4-------D2
              2H1S1T6-------D2
)")
                .substr(1),
            /* print_timestamp = */ true);

  EXPECT_TRUE(mtf.Insert(10));
  CheckTree(mtf,
            std::string(R"(
3H3S4T5-------1H1S1T4-------D2
              2H2S2T6-------D2
                            10H1S1T7------D3
)")
                .substr(1),
            /* print_timestamp = */ true);

  EXPECT_TRUE(mtf.ValueFromRank(1, &value));
  EXPECT_EQ(10u, value);
}

TEST(MoveToFront, Remove) {
  MoveToFrontTester mtf;

  EXPECT_FALSE(mtf.Remove(1));
  EXPECT_EQ(0u, mtf.GetTotalNodeCount());

  EXPECT_TRUE(mtf.Insert(1));
  EXPECT_TRUE(mtf.Insert(2));
  EXPECT_TRUE(mtf.Insert(3));

  CheckTree(mtf,
            std::string(R"(
2H2S3T2-------1H1S1T1-------D2
              3H1S1T3-------D2
)")
                .substr(1),
            /* print_timestamp = */ true);

  EXPECT_EQ(1u, mtf.GetNodeHandle(1));
  EXPECT_EQ(3u, mtf.GetTotalNodeCount());
  EXPECT_TRUE(mtf.Remove(1));
  EXPECT_EQ(3u, mtf.GetTotalNodeCount());

  CheckTree(mtf,
            std::string(R"(
2H2S2T2-------D1
              3H1S1T3-------D2
)")
                .substr(1),
            /* print_timestamp = */ true);

  uint32_t value = 0;
  EXPECT_TRUE(mtf.ValueFromRank(2, &value));
  EXPECT_EQ(2u, value);

  CheckTree(mtf,
            std::string(R"(
3H2S2T3-------D1
              2H1S1T4-------D2
)")
                .substr(1),
            /* print_timestamp = */ true);

  EXPECT_TRUE(mtf.Insert(1));
  EXPECT_EQ(1u, mtf.GetNodeHandle(1));
  EXPECT_EQ(3u, mtf.GetTotalNodeCount());
}

TEST(MoveToFront, LargerScale) {
  MoveToFrontTester mtf;
  uint32_t value = 0;
  uint32_t rank = 0;

  for (uint32_t i = 1; i < 1000; ++i) {
    ASSERT_TRUE(mtf.Insert(i));
    ASSERT_EQ(i, mtf.GetSize());

    ASSERT_TRUE(mtf.RankFromValue(i, &rank));
    ASSERT_EQ(1u, rank);

    ASSERT_TRUE(mtf.ValueFromRank(1, &value));
    ASSERT_EQ(i, value);
  }

  ASSERT_TRUE(mtf.ValueFromRank(999, &value));
  ASSERT_EQ(1u, value);

  ASSERT_TRUE(mtf.ValueFromRank(999, &value));
  ASSERT_EQ(2u, value);

  ASSERT_TRUE(mtf.ValueFromRank(999, &value));
  ASSERT_EQ(3u, value);

  ASSERT_TRUE(mtf.ValueFromRank(999, &value));
  ASSERT_EQ(4u, value);

  ASSERT_TRUE(mtf.ValueFromRank(999, &value));
  ASSERT_EQ(5u, value);

  ASSERT_TRUE(mtf.ValueFromRank(999, &value));
  ASSERT_EQ(6u, value);

  ASSERT_TRUE(mtf.ValueFromRank(101, &value));
  ASSERT_EQ(905u, value);

  ASSERT_TRUE(mtf.ValueFromRank(101, &value));
  ASSERT_EQ(906u, value);

  ASSERT_TRUE(mtf.ValueFromRank(101, &value));
  ASSERT_EQ(907u, value);

  ASSERT_TRUE(mtf.ValueFromRank(201, &value));
  ASSERT_EQ(805u, value);

  ASSERT_TRUE(mtf.ValueFromRank(201, &value));
  ASSERT_EQ(806u, value);

  ASSERT_TRUE(mtf.ValueFromRank(201, &value));
  ASSERT_EQ(807u, value);

  ASSERT_TRUE(mtf.ValueFromRank(301, &value));
  ASSERT_EQ(705u, value);

  ASSERT_TRUE(mtf.ValueFromRank(301, &value));
  ASSERT_EQ(706u, value);

  ASSERT_TRUE(mtf.ValueFromRank(301, &value));
  ASSERT_EQ(707u, value);

  ASSERT_TRUE(mtf.RankFromValue(605, &rank));
  ASSERT_EQ(401u, rank);

  ASSERT_TRUE(mtf.RankFromValue(606, &rank));
  ASSERT_EQ(401u, rank);

  ASSERT_TRUE(mtf.RankFromValue(607, &rank));
  ASSERT_EQ(401u, rank);

  ASSERT_TRUE(mtf.ValueFromRank(1, &value));
  ASSERT_EQ(607u, value);

  ASSERT_TRUE(mtf.ValueFromRank(2, &value));
  ASSERT_EQ(606u, value);

  ASSERT_TRUE(mtf.ValueFromRank(3, &value));
  ASSERT_EQ(605u, value);

  ASSERT_TRUE(mtf.ValueFromRank(4, &value));
  ASSERT_EQ(707u, value);

  ASSERT_TRUE(mtf.ValueFromRank(5, &value));
  ASSERT_EQ(706u, value);

  ASSERT_TRUE(mtf.ValueFromRank(6, &value));
  ASSERT_EQ(705u, value);

  ASSERT_TRUE(mtf.ValueFromRank(7, &value));
  ASSERT_EQ(807u, value);

  ASSERT_TRUE(mtf.ValueFromRank(8, &value));
  ASSERT_EQ(806u, value);

  ASSERT_TRUE(mtf.ValueFromRank(9, &value));
  ASSERT_EQ(805u, value);

  ASSERT_TRUE(mtf.ValueFromRank(10, &value));
  ASSERT_EQ(907u, value);

  ASSERT_TRUE(mtf.ValueFromRank(11, &value));
  ASSERT_EQ(906u, value);

  ASSERT_TRUE(mtf.ValueFromRank(12, &value));
  ASSERT_EQ(905u, value);

  ASSERT_TRUE(mtf.ValueFromRank(13, &value));
  ASSERT_EQ(6u, value);

  ASSERT_TRUE(mtf.ValueFromRank(14, &value));
  ASSERT_EQ(5u, value);

  ASSERT_TRUE(mtf.ValueFromRank(15, &value));
  ASSERT_EQ(4u, value);

  ASSERT_TRUE(mtf.ValueFromRank(16, &value));
  ASSERT_EQ(3u, value);

  ASSERT_TRUE(mtf.ValueFromRank(17, &value));
  ASSERT_EQ(2u, value);

  ASSERT_TRUE(mtf.ValueFromRank(18, &value));
  ASSERT_EQ(1u, value);

  ASSERT_TRUE(mtf.ValueFromRank(19, &value));
  ASSERT_EQ(999u, value);

  ASSERT_TRUE(mtf.ValueFromRank(20, &value));
  ASSERT_EQ(998u, value);

  ASSERT_TRUE(mtf.ValueFromRank(21, &value));
  ASSERT_EQ(997u, value);

  ASSERT_TRUE(mtf.RankFromValue(997, &rank));
  ASSERT_EQ(1u, rank);

  ASSERT_TRUE(mtf.RankFromValue(998, &rank));
  ASSERT_EQ(2u, rank);

  ASSERT_TRUE(mtf.RankFromValue(996, &rank));
  ASSERT_EQ(22u, rank);

  ASSERT_TRUE(mtf.Remove(995));

  ASSERT_TRUE(mtf.RankFromValue(994, &rank));
  ASSERT_EQ(23u, rank);

  for (uint32_t i = 10; i < 1000; ++i) {
    if (i != 995) {
      ASSERT_TRUE(mtf.Remove(i));
    } else {
      ASSERT_FALSE(mtf.Remove(i));
    }
  }

  CheckTree(mtf,
            std::string(R"(
6H4S9T1029----8H2S3T8-------7H1S1T7-------D3
                            9H1S1T9-------D3
              2H3S5T1033----4H2S3T1031----5H1S1T1030----D4
                                          3H1S1T1032----D4
                            1H1S1T1034----D3
)")
                .substr(1),
            /* print_timestamp = */ true);

  ASSERT_TRUE(mtf.Insert(1000));
  ASSERT_TRUE(mtf.ValueFromRank(1, &value));
  ASSERT_EQ(1000u, value);
}

}  // namespace
}  // namespace comp
}  // namespace spvtools
