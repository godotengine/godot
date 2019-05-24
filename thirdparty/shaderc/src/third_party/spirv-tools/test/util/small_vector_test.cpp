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

#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "source/util/small_vector.h"

namespace spvtools {
namespace utils {
namespace {

using SmallVectorTest = ::testing::Test;

TEST(SmallVectorTest, Initialize_default) {
  SmallVector<uint32_t, 2> vec;

  EXPECT_TRUE(vec.empty());
  EXPECT_EQ(vec.size(), 0);
  EXPECT_EQ(vec.begin(), vec.end());
}

TEST(SmallVectorTest, Initialize_list1) {
  SmallVector<uint32_t, 2> vec = {0, 1, 2, 3};

  EXPECT_FALSE(vec.empty());
  EXPECT_EQ(vec.size(), 4);

  uint32_t result[] = {0, 1, 2, 3};
  for (uint32_t i = 0; i < vec.size(); ++i) {
    EXPECT_EQ(vec[i], result[i]);
  }
}

TEST(SmallVectorTest, Initialize_list2) {
  SmallVector<uint32_t, 6> vec = {0, 1, 2, 3};

  EXPECT_FALSE(vec.empty());
  EXPECT_EQ(vec.size(), 4);

  uint32_t result[] = {0, 1, 2, 3};
  for (uint32_t i = 0; i < vec.size(); ++i) {
    EXPECT_EQ(vec[i], result[i]);
  }
}

TEST(SmallVectorTest, Initialize_copy1) {
  SmallVector<uint32_t, 6> vec1 = {0, 1, 2, 3};
  SmallVector<uint32_t, 6> vec2(vec1);

  EXPECT_EQ(vec2.size(), 4);

  uint32_t result[] = {0, 1, 2, 3};
  for (uint32_t i = 0; i < vec2.size(); ++i) {
    EXPECT_EQ(vec2[i], result[i]);
  }

  EXPECT_EQ(vec1, vec2);
}

TEST(SmallVectorTest, Initialize_copy2) {
  SmallVector<uint32_t, 2> vec1 = {0, 1, 2, 3};
  SmallVector<uint32_t, 2> vec2(vec1);

  EXPECT_EQ(vec2.size(), 4);

  uint32_t result[] = {0, 1, 2, 3};
  for (uint32_t i = 0; i < vec2.size(); ++i) {
    EXPECT_EQ(vec2[i], result[i]);
  }

  EXPECT_EQ(vec1, vec2);
}

TEST(SmallVectorTest, Initialize_copy_vec1) {
  std::vector<uint32_t> vec1 = {0, 1, 2, 3};
  SmallVector<uint32_t, 6> vec2(vec1);

  EXPECT_EQ(vec2.size(), 4);

  uint32_t result[] = {0, 1, 2, 3};
  for (uint32_t i = 0; i < vec2.size(); ++i) {
    EXPECT_EQ(vec2[i], result[i]);
  }

  EXPECT_EQ(vec1, vec2);
}

TEST(SmallVectorTest, Initialize_copy_vec2) {
  std::vector<uint32_t> vec1 = {0, 1, 2, 3};
  SmallVector<uint32_t, 2> vec2(vec1);

  EXPECT_EQ(vec2.size(), 4);

  uint32_t result[] = {0, 1, 2, 3};
  for (uint32_t i = 0; i < vec2.size(); ++i) {
    EXPECT_EQ(vec2[i], result[i]);
  }

  EXPECT_EQ(vec1, vec2);
}

TEST(SmallVectorTest, Initialize_move1) {
  SmallVector<uint32_t, 6> vec1 = {0, 1, 2, 3};
  SmallVector<uint32_t, 6> vec2(std::move(vec1));

  EXPECT_EQ(vec2.size(), 4);

  uint32_t result[] = {0, 1, 2, 3};
  for (uint32_t i = 0; i < vec2.size(); ++i) {
    EXPECT_EQ(vec2[i], result[i]);
  }
  EXPECT_TRUE(vec1.empty());
}

TEST(SmallVectorTest, Initialize_move2) {
  SmallVector<uint32_t, 2> vec1 = {0, 1, 2, 3};
  SmallVector<uint32_t, 2> vec2(std::move(vec1));

  EXPECT_EQ(vec2.size(), 4);

  uint32_t result[] = {0, 1, 2, 3};
  for (uint32_t i = 0; i < vec2.size(); ++i) {
    EXPECT_EQ(vec2[i], result[i]);
  }
  EXPECT_TRUE(vec1.empty());
}

TEST(SmallVectorTest, Initialize_move_vec1) {
  std::vector<uint32_t> vec1 = {0, 1, 2, 3};
  SmallVector<uint32_t, 6> vec2(std::move(vec1));

  EXPECT_EQ(vec2.size(), 4);

  uint32_t result[] = {0, 1, 2, 3};
  for (uint32_t i = 0; i < vec2.size(); ++i) {
    EXPECT_EQ(vec2[i], result[i]);
  }
  EXPECT_TRUE(vec1.empty());
}

TEST(SmallVectorTest, Initialize_move_vec2) {
  std::vector<uint32_t> vec1 = {0, 1, 2, 3};
  SmallVector<uint32_t, 2> vec2(std::move(vec1));

  EXPECT_EQ(vec2.size(), 4);

  uint32_t result[] = {0, 1, 2, 3};
  for (uint32_t i = 0; i < vec2.size(); ++i) {
    EXPECT_EQ(vec2[i], result[i]);
  }
  EXPECT_TRUE(vec1.empty());
}

TEST(SmallVectorTest, Initialize_iterators1) {
  SmallVector<uint32_t, 2> vec = {0, 1, 2, 3};

  EXPECT_EQ(vec.size(), 4);
  uint32_t result[] = {0, 1, 2, 3};

  uint32_t i = 0;
  for (uint32_t p : vec) {
    EXPECT_EQ(p, result[i]);
    i++;
  }
}

TEST(SmallVectorTest, Initialize_iterators2) {
  SmallVector<uint32_t, 6> vec = {0, 1, 2, 3};

  EXPECT_EQ(vec.size(), 4);
  uint32_t result[] = {0, 1, 2, 3};

  uint32_t i = 0;
  for (uint32_t p : vec) {
    EXPECT_EQ(p, result[i]);
    i++;
  }
}

TEST(SmallVectorTest, Initialize_iterators3) {
  SmallVector<uint32_t, 2> vec = {0, 1, 2, 3};

  EXPECT_EQ(vec.size(), 4);
  uint32_t result[] = {0, 1, 2, 3};

  uint32_t i = 0;
  for (SmallVector<uint32_t, 2>::iterator it = vec.begin(); it != vec.end();
       ++it) {
    EXPECT_EQ(*it, result[i]);
    i++;
  }
}

TEST(SmallVectorTest, Initialize_iterators4) {
  SmallVector<uint32_t, 6> vec = {0, 1, 2, 3};

  EXPECT_EQ(vec.size(), 4);
  uint32_t result[] = {0, 1, 2, 3};

  uint32_t i = 0;
  for (SmallVector<uint32_t, 6>::iterator it = vec.begin(); it != vec.end();
       ++it) {
    EXPECT_EQ(*it, result[i]);
    i++;
  }
}

TEST(SmallVectorTest, Initialize_iterators_write1) {
  SmallVector<uint32_t, 6> vec = {0, 1, 2, 3};

  EXPECT_EQ(vec.size(), 4);
  for (SmallVector<uint32_t, 6>::iterator it = vec.begin(); it != vec.end();
       ++it) {
    *it *= 2;
  }

  uint32_t result[] = {0, 2, 4, 6};

  uint32_t i = 0;
  for (SmallVector<uint32_t, 6>::iterator it = vec.begin(); it != vec.end();
       ++it) {
    EXPECT_EQ(*it, result[i]);
    i++;
  }
}

TEST(SmallVectorTest, Initialize_iterators_write2) {
  SmallVector<uint32_t, 2> vec = {0, 1, 2, 3};

  EXPECT_EQ(vec.size(), 4);
  for (SmallVector<uint32_t, 2>::iterator it = vec.begin(); it != vec.end();
       ++it) {
    *it *= 2;
  }

  uint32_t result[] = {0, 2, 4, 6};

  uint32_t i = 0;
  for (SmallVector<uint32_t, 2>::iterator it = vec.begin(); it != vec.end();
       ++it) {
    EXPECT_EQ(*it, result[i]);
    i++;
  }
}

TEST(SmallVectorTest, Initialize_front) {
  SmallVector<uint32_t, 2> vec = {0, 1, 2, 3};

  EXPECT_EQ(vec.front(), 0);
  for (SmallVector<uint32_t, 2>::iterator it = vec.begin(); it != vec.end();
       ++it) {
    *it += 2;
  }
  EXPECT_EQ(vec.front(), 2);
}

TEST(SmallVectorTest, Erase_element_front1) {
  SmallVector<uint32_t, 2> vec = {0, 1, 2, 3};

  EXPECT_EQ(vec.front(), 0);
  EXPECT_EQ(vec.size(), 4);
  vec.erase(vec.begin());
  EXPECT_EQ(vec.front(), 1);
  EXPECT_EQ(vec.size(), 3);
}

TEST(SmallVectorTest, Erase_element_front2) {
  SmallVector<uint32_t, 6> vec = {0, 1, 2, 3};

  EXPECT_EQ(vec.front(), 0);
  EXPECT_EQ(vec.size(), 4);
  vec.erase(vec.begin());
  EXPECT_EQ(vec.front(), 1);
  EXPECT_EQ(vec.size(), 3);
}

TEST(SmallVectorTest, Erase_element_back1) {
  SmallVector<uint32_t, 2> vec = {0, 1, 2, 3};
  SmallVector<uint32_t, 2> result = {0, 1, 2};

  EXPECT_EQ(vec[3], 3);
  EXPECT_EQ(vec.size(), 4);
  vec.erase(vec.begin() + 3);
  EXPECT_EQ(vec.size(), 3);
  EXPECT_EQ(vec, result);
}

TEST(SmallVectorTest, Erase_element_back2) {
  SmallVector<uint32_t, 6> vec = {0, 1, 2, 3};
  SmallVector<uint32_t, 6> result = {0, 1, 2};

  EXPECT_EQ(vec[3], 3);
  EXPECT_EQ(vec.size(), 4);
  vec.erase(vec.begin() + 3);
  EXPECT_EQ(vec.size(), 3);
  EXPECT_EQ(vec, result);
}

TEST(SmallVectorTest, Erase_element_middle1) {
  SmallVector<uint32_t, 2> vec = {0, 1, 2, 3};
  SmallVector<uint32_t, 2> result = {0, 1, 3};

  EXPECT_EQ(vec.size(), 4);
  vec.erase(vec.begin() + 2);
  EXPECT_EQ(vec.size(), 3);
  EXPECT_EQ(vec, result);
}

TEST(SmallVectorTest, Erase_element_middle2) {
  SmallVector<uint32_t, 6> vec = {0, 1, 2, 3};
  SmallVector<uint32_t, 6> result = {0, 1, 3};

  EXPECT_EQ(vec.size(), 4);
  vec.erase(vec.begin() + 2);
  EXPECT_EQ(vec.size(), 3);
  EXPECT_EQ(vec, result);
}

TEST(SmallVectorTest, Erase_range_1) {
  SmallVector<uint32_t, 6> vec = {0, 1, 2, 3};
  SmallVector<uint32_t, 6> result = {};

  EXPECT_EQ(vec.size(), 4);
  vec.erase(vec.begin(), vec.end());
  EXPECT_EQ(vec.size(), 0);
  EXPECT_EQ(vec, result);
}

TEST(SmallVectorTest, Erase_range_2) {
  SmallVector<uint32_t, 2> vec = {0, 1, 2, 3};
  SmallVector<uint32_t, 2> result = {};

  EXPECT_EQ(vec.size(), 4);
  vec.erase(vec.begin(), vec.end());
  EXPECT_EQ(vec.size(), 0);
  EXPECT_EQ(vec, result);
}

TEST(SmallVectorTest, Erase_range_3) {
  SmallVector<uint32_t, 6> vec = {0, 1, 2, 3};
  SmallVector<uint32_t, 6> result = {2, 3};

  EXPECT_EQ(vec.size(), 4);
  vec.erase(vec.begin(), vec.begin() + 2);
  EXPECT_EQ(vec.size(), 2);
  EXPECT_EQ(vec, result);
}

TEST(SmallVectorTest, Erase_range_4) {
  SmallVector<uint32_t, 2> vec = {0, 1, 2, 3};
  SmallVector<uint32_t, 2> result = {2, 3};

  EXPECT_EQ(vec.size(), 4);
  vec.erase(vec.begin(), vec.begin() + 2);
  EXPECT_EQ(vec.size(), 2);
  EXPECT_EQ(vec, result);
}

TEST(SmallVectorTest, Erase_range_5) {
  SmallVector<uint32_t, 6> vec = {0, 1, 2, 3};
  SmallVector<uint32_t, 6> result = {0, 3};

  EXPECT_EQ(vec.size(), 4);
  vec.erase(vec.begin() + 1, vec.begin() + 3);
  EXPECT_EQ(vec.size(), 2);
  EXPECT_EQ(vec, result);
}

TEST(SmallVectorTest, Erase_range_6) {
  SmallVector<uint32_t, 2> vec = {0, 1, 2, 3};
  SmallVector<uint32_t, 2> result = {0, 3};

  EXPECT_EQ(vec.size(), 4);
  vec.erase(vec.begin() + 1, vec.begin() + 3);
  EXPECT_EQ(vec.size(), 2);
  EXPECT_EQ(vec, result);
}

TEST(SmallVectorTest, Push_back) {
  SmallVector<uint32_t, 2> vec;
  SmallVector<uint32_t, 2> result = {0, 1, 2, 3};

  EXPECT_EQ(vec.size(), 0);
  vec.push_back(0);
  EXPECT_EQ(vec.size(), 1);
  vec.push_back(1);
  EXPECT_EQ(vec.size(), 2);
  vec.push_back(2);
  EXPECT_EQ(vec.size(), 3);
  vec.push_back(3);
  EXPECT_EQ(vec.size(), 4);
  EXPECT_EQ(vec, result);
}

TEST(SmallVectorTest, Emplace_back) {
  SmallVector<uint32_t, 2> vec;
  SmallVector<uint32_t, 2> result = {0, 1, 2, 3};

  EXPECT_EQ(vec.size(), 0);
  vec.emplace_back(0);
  EXPECT_EQ(vec.size(), 1);
  vec.emplace_back(1);
  EXPECT_EQ(vec.size(), 2);
  vec.emplace_back(2);
  EXPECT_EQ(vec.size(), 3);
  vec.emplace_back(3);
  EXPECT_EQ(vec.size(), 4);
  EXPECT_EQ(vec, result);
}

TEST(SmallVectorTest, Clear) {
  SmallVector<uint32_t, 2> vec = {0, 1, 2, 3};
  SmallVector<uint32_t, 2> result = {};

  EXPECT_EQ(vec.size(), 4);
  vec.clear();
  EXPECT_EQ(vec.size(), 0);
  EXPECT_EQ(vec, result);
}

TEST(SmallVectorTest, Insert1) {
  SmallVector<uint32_t, 2> vec = {};
  SmallVector<uint32_t, 2> insert_values = {10, 11};
  SmallVector<uint32_t, 2> result = {10, 11};

  EXPECT_EQ(vec.size(), 0);
  auto ret =
      vec.insert(vec.begin(), insert_values.begin(), insert_values.end());
  EXPECT_EQ(vec.size(), 2);
  EXPECT_EQ(vec, result);
  EXPECT_EQ(*ret, 10);
}

TEST(SmallVectorTest, Insert2) {
  SmallVector<uint32_t, 2> vec = {};
  SmallVector<uint32_t, 2> insert_values = {10, 11, 12};
  SmallVector<uint32_t, 2> result = {10, 11, 12};

  EXPECT_EQ(vec.size(), 0);
  auto ret =
      vec.insert(vec.begin(), insert_values.begin(), insert_values.end());
  EXPECT_EQ(vec.size(), 3);
  EXPECT_EQ(vec, result);
  EXPECT_EQ(*ret, 10);
}

TEST(SmallVectorTest, Insert3) {
  SmallVector<uint32_t, 2> vec = {0};
  SmallVector<uint32_t, 2> insert_values = {10, 11, 12};
  SmallVector<uint32_t, 2> result = {10, 11, 12, 0};

  EXPECT_EQ(vec.size(), 1);
  auto ret =
      vec.insert(vec.begin(), insert_values.begin(), insert_values.end());
  EXPECT_EQ(vec.size(), 4);
  EXPECT_EQ(vec, result);
  EXPECT_EQ(*ret, 10);
}

TEST(SmallVectorTest, Insert4) {
  SmallVector<uint32_t, 6> vec = {0};
  SmallVector<uint32_t, 6> insert_values = {10, 11, 12};
  SmallVector<uint32_t, 6> result = {10, 11, 12, 0};

  EXPECT_EQ(vec.size(), 1);
  auto ret =
      vec.insert(vec.begin(), insert_values.begin(), insert_values.end());
  EXPECT_EQ(vec.size(), 4);
  EXPECT_EQ(vec, result);
  EXPECT_EQ(*ret, 10);
}

TEST(SmallVectorTest, Insert5) {
  SmallVector<uint32_t, 2> vec = {0, 1, 2};
  SmallVector<uint32_t, 2> insert_values = {10, 11, 12};
  SmallVector<uint32_t, 2> result = {0, 1, 2, 10, 11, 12};

  EXPECT_EQ(vec.size(), 3);
  auto ret = vec.insert(vec.end(), insert_values.begin(), insert_values.end());
  EXPECT_EQ(vec.size(), 6);
  EXPECT_EQ(vec, result);
  EXPECT_EQ(*ret, 10);
}

TEST(SmallVectorTest, Insert6) {
  SmallVector<uint32_t, 6> vec = {0, 1, 2};
  SmallVector<uint32_t, 6> insert_values = {10, 11, 12};
  SmallVector<uint32_t, 6> result = {0, 1, 2, 10, 11, 12};

  EXPECT_EQ(vec.size(), 3);
  auto ret = vec.insert(vec.end(), insert_values.begin(), insert_values.end());
  EXPECT_EQ(vec.size(), 6);
  EXPECT_EQ(vec, result);
  EXPECT_EQ(*ret, 10);
}

TEST(SmallVectorTest, Insert7) {
  SmallVector<uint32_t, 2> vec = {0, 1, 2};
  SmallVector<uint32_t, 2> insert_values = {10, 11, 12};
  SmallVector<uint32_t, 2> result = {0, 10, 11, 12, 1, 2};

  EXPECT_EQ(vec.size(), 3);
  auto ret =
      vec.insert(vec.begin() + 1, insert_values.begin(), insert_values.end());
  EXPECT_EQ(vec.size(), 6);
  EXPECT_EQ(vec, result);
  EXPECT_EQ(*ret, 10);
}

TEST(SmallVectorTest, Insert8) {
  SmallVector<uint32_t, 6> vec = {0, 1, 2};
  SmallVector<uint32_t, 6> insert_values = {10, 11, 12};
  SmallVector<uint32_t, 6> result = {0, 10, 11, 12, 1, 2};

  EXPECT_EQ(vec.size(), 3);
  auto ret =
      vec.insert(vec.begin() + 1, insert_values.begin(), insert_values.end());
  EXPECT_EQ(vec.size(), 6);
  EXPECT_EQ(vec, result);
  EXPECT_EQ(*ret, 10);
}

TEST(SmallVectorTest, Resize1) {
  SmallVector<uint32_t, 2> vec = {0, 1, 2};
  SmallVector<uint32_t, 2> result = {0, 1, 2, 10, 10, 10};

  EXPECT_EQ(vec.size(), 3);
  vec.resize(6, 10);
  EXPECT_EQ(vec.size(), 6);
  EXPECT_EQ(vec, result);
}

TEST(SmallVectorTest, Resize2) {
  SmallVector<uint32_t, 8> vec = {0, 1, 2};
  SmallVector<uint32_t, 8> result = {0, 1, 2, 10, 10, 10};

  EXPECT_EQ(vec.size(), 3);
  vec.resize(6, 10);
  EXPECT_EQ(vec.size(), 6);
  EXPECT_EQ(vec, result);
}

TEST(SmallVectorTest, Resize3) {
  SmallVector<uint32_t, 4> vec = {0, 1, 2};
  SmallVector<uint32_t, 4> result = {0, 1, 2, 10, 10, 10};

  EXPECT_EQ(vec.size(), 3);
  vec.resize(6, 10);
  EXPECT_EQ(vec.size(), 6);
  EXPECT_EQ(vec, result);
}

TEST(SmallVectorTest, Resize4) {
  SmallVector<uint32_t, 4> vec = {0, 1, 2, 10, 10, 10};
  SmallVector<uint32_t, 4> result = {0, 1, 2};

  EXPECT_EQ(vec.size(), 6);
  vec.resize(3, 10);
  EXPECT_EQ(vec.size(), 3);
  EXPECT_EQ(vec, result);
}

TEST(SmallVectorTest, Resize5) {
  SmallVector<uint32_t, 2> vec = {0, 1, 2, 10, 10, 10};
  SmallVector<uint32_t, 2> result = {0, 1, 2};

  EXPECT_EQ(vec.size(), 6);
  vec.resize(3, 10);
  EXPECT_EQ(vec.size(), 3);
  EXPECT_EQ(vec, result);
}

TEST(SmallVectorTest, Resize6) {
  SmallVector<uint32_t, 8> vec = {0, 1, 2, 10, 10, 10};
  SmallVector<uint32_t, 8> result = {0, 1, 2};

  EXPECT_EQ(vec.size(), 6);
  vec.resize(3, 10);
  EXPECT_EQ(vec.size(), 3);
  EXPECT_EQ(vec, result);
}

}  // namespace
}  // namespace utils
}  // namespace spvtools
