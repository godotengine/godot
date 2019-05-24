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

#include <memory>
#include <vector>

#include "gmock/gmock.h"

#include "source/opt/iterator.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace opt {
namespace {

using ::testing::ContainerEq;

TEST(Iterator, IncrementDeref) {
  const int count = 100;
  std::vector<std::unique_ptr<int>> data;
  for (int i = 0; i < count; ++i) {
    data.emplace_back(new int(i));
  }

  UptrVectorIterator<int> it(&data, data.begin());
  UptrVectorIterator<int> end(&data, data.end());

  EXPECT_EQ(*data[0], *it);
  for (int i = 1; i < count; ++i) {
    EXPECT_NE(end, it);
    EXPECT_EQ(*data[i], *(++it));
  }
  EXPECT_EQ(end, ++it);
}

TEST(Iterator, DecrementDeref) {
  const int count = 100;
  std::vector<std::unique_ptr<int>> data;
  for (int i = 0; i < count; ++i) {
    data.emplace_back(new int(i));
  }

  UptrVectorIterator<int> begin(&data, data.begin());
  UptrVectorIterator<int> it(&data, data.end());

  for (int i = count - 1; i >= 0; --i) {
    EXPECT_NE(begin, it);
    EXPECT_EQ(*data[i], *(--it));
  }
  EXPECT_EQ(begin, it);
}

TEST(Iterator, PostIncrementDeref) {
  const int count = 100;
  std::vector<std::unique_ptr<int>> data;
  for (int i = 0; i < count; ++i) {
    data.emplace_back(new int(i));
  }

  UptrVectorIterator<int> it(&data, data.begin());
  UptrVectorIterator<int> end(&data, data.end());

  for (int i = 0; i < count; ++i) {
    EXPECT_NE(end, it);
    EXPECT_EQ(*data[i], *(it++));
  }
  EXPECT_EQ(end, it);
}

TEST(Iterator, PostDecrementDeref) {
  const int count = 100;
  std::vector<std::unique_ptr<int>> data;
  for (int i = 0; i < count; ++i) {
    data.emplace_back(new int(i));
  }

  UptrVectorIterator<int> begin(&data, data.begin());
  UptrVectorIterator<int> end(&data, data.end());
  UptrVectorIterator<int> it(&data, data.end());

  EXPECT_EQ(end, it--);
  for (int i = count - 1; i >= 1; --i) {
    EXPECT_EQ(*data[i], *(it--));
  }
  // Decrementing .begin() is undefined behavior.
  EXPECT_EQ(*data[0], *it);
}

TEST(Iterator, Access) {
  const int count = 100;
  std::vector<std::unique_ptr<int>> data;
  for (int i = 0; i < count; ++i) {
    data.emplace_back(new int(i));
  }

  UptrVectorIterator<int> it(&data, data.begin());

  for (int i = 0; i < count; ++i) EXPECT_EQ(*data[i], it[i]);
}

TEST(Iterator, Comparison) {
  const int count = 100;
  std::vector<std::unique_ptr<int>> data;
  for (int i = 0; i < count; ++i) {
    data.emplace_back(new int(i));
  }

  UptrVectorIterator<int> it(&data, data.begin());
  UptrVectorIterator<int> end(&data, data.end());

  for (int i = 0; i < count; ++i, ++it) EXPECT_TRUE(it < end);
  EXPECT_EQ(end, it);
}

TEST(Iterator, InsertBeginEnd) {
  const int count = 100;

  std::vector<std::unique_ptr<int>> data;
  std::vector<int> expected;
  std::vector<int> actual;

  for (int i = 0; i < count; ++i) {
    data.emplace_back(new int(i));
    expected.push_back(i);
  }

  // Insert at the beginning
  expected.insert(expected.begin(), -100);
  UptrVectorIterator<int> begin(&data, data.begin());
  auto insert_point = begin.InsertBefore(MakeUnique<int>(-100));
  for (int i = 0; i < count + 1; ++i) {
    actual.push_back(*(insert_point++));
  }
  EXPECT_THAT(actual, ContainerEq(expected));

  // Insert at the end
  expected.push_back(-42);
  expected.push_back(-36);
  expected.push_back(-77);
  UptrVectorIterator<int> end(&data, data.end());
  end = end.InsertBefore(MakeUnique<int>(-77));
  end = end.InsertBefore(MakeUnique<int>(-36));
  end = end.InsertBefore(MakeUnique<int>(-42));

  actual.clear();
  begin = UptrVectorIterator<int>(&data, data.begin());
  for (int i = 0; i < count + 4; ++i) {
    actual.push_back(*(begin++));
  }
  EXPECT_THAT(actual, ContainerEq(expected));
}

TEST(Iterator, InsertMiddle) {
  const int count = 100;

  std::vector<std::unique_ptr<int>> data;
  std::vector<int> expected;
  std::vector<int> actual;

  for (int i = 0; i < count; ++i) {
    data.emplace_back(new int(i));
    expected.push_back(i);
  }

  const int insert_pos = 42;
  expected.insert(expected.begin() + insert_pos, -100);
  expected.insert(expected.begin() + insert_pos, -42);

  UptrVectorIterator<int> it(&data, data.begin());
  for (int i = 0; i < insert_pos; ++i) ++it;
  it = it.InsertBefore(MakeUnique<int>(-100));
  it = it.InsertBefore(MakeUnique<int>(-42));
  auto begin = UptrVectorIterator<int>(&data, data.begin());
  for (int i = 0; i < count + 2; ++i) {
    actual.push_back(*(begin++));
  }
  EXPECT_THAT(actual, ContainerEq(expected));
}

TEST(IteratorRange, Interface) {
  const uint32_t count = 100;

  std::vector<std::unique_ptr<uint32_t>> data;

  for (uint32_t i = 0; i < count; ++i) {
    data.emplace_back(new uint32_t(i));
  }

  auto b = UptrVectorIterator<uint32_t>(&data, data.begin());
  auto e = UptrVectorIterator<uint32_t>(&data, data.end());
  auto range = IteratorRange<decltype(b)>(b, e);

  EXPECT_EQ(b, range.begin());
  EXPECT_EQ(e, range.end());
  EXPECT_FALSE(range.empty());
  EXPECT_EQ(count, range.size());
  EXPECT_EQ(0u, *range.begin());
  EXPECT_EQ(99u, *(--range.end()));

  // IteratorRange itself is immutable.
  ++b, --e;
  EXPECT_EQ(count, range.size());
  ++range.begin(), --range.end();
  EXPECT_EQ(count, range.size());
}

TEST(Iterator, FilterIterator) {
  struct Placeholder {
    int val;
  };
  std::vector<Placeholder> data = {{1}, {2}, {3}, {4}, {5},
                                   {6}, {7}, {8}, {9}, {10}};

  // Predicate to only consider odd values.
  struct Predicate {
    bool operator()(const Placeholder& data) { return data.val % 2; }
  };
  Predicate pred;

  auto filter_range = MakeFilterIteratorRange(data.begin(), data.end(), pred);

  EXPECT_EQ(filter_range.begin().Get(), data.begin());
  EXPECT_EQ(filter_range.end(), filter_range.begin().GetEnd());

  for (Placeholder& data : filter_range) {
    EXPECT_EQ(data.val % 2, 1);
  }

  for (auto it = filter_range.begin(); it != filter_range.end(); it++) {
    EXPECT_EQ(it->val % 2, 1);
    EXPECT_EQ((*it).val % 2, 1);
  }

  for (auto it = filter_range.begin(); it != filter_range.end(); ++it) {
    EXPECT_EQ(it->val % 2, 1);
    EXPECT_EQ((*it).val % 2, 1);
  }

  EXPECT_EQ(MakeFilterIterator(data.begin(), data.end(), pred).Get(),
            data.begin());
  EXPECT_EQ(MakeFilterIterator(data.end(), data.end(), pred).Get(), data.end());
  EXPECT_EQ(MakeFilterIterator(data.begin(), data.end(), pred).GetEnd(),
            MakeFilterIterator(data.end(), data.end(), pred));
  EXPECT_NE(MakeFilterIterator(data.begin(), data.end(), pred),
            MakeFilterIterator(data.end(), data.end(), pred));

  // Empty range: no values satisfies the predicate.
  auto empty_range = MakeFilterIteratorRange(
      data.begin(), data.end(),
      [](const Placeholder& data) { return data.val > 10; });
  EXPECT_EQ(empty_range.begin(), empty_range.end());
}

}  // namespace
}  // namespace opt
}  // namespace spvtools
