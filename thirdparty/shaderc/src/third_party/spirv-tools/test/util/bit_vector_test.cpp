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

#include <vector>

#include "gmock/gmock.h"

#include "source/util/bit_vector.h"

namespace spvtools {
namespace utils {
namespace {

using BitVectorTest = ::testing::Test;

TEST(BitVectorTest, Initialize) {
  BitVector bvec;

  // Checks that all values are 0.  Also tests checking a  bit past the end of
  // the vector containing the bits.
  for (int i = 1; i < 10000; i *= 2) {
    EXPECT_FALSE(bvec.Get(i));
  }
}

TEST(BitVectorTest, Set) {
  BitVector bvec;

  // Since 10,000 is larger than the initial size, this tests the resizing
  // code.
  for (int i = 3; i < 10000; i *= 2) {
    bvec.Set(i);
  }

  // Check that bits that were not set are 0.
  for (int i = 1; i < 10000; i *= 2) {
    EXPECT_FALSE(bvec.Get(i));
  }

  // Check that bits that were set are 1.
  for (int i = 3; i < 10000; i *= 2) {
    EXPECT_TRUE(bvec.Get(i));
  }
}

TEST(BitVectorTest, SetReturnValue) {
  BitVector bvec;

  // Make sure |Set| returns false when the bit was not set.
  for (int i = 3; i < 10000; i *= 2) {
    EXPECT_FALSE(bvec.Set(i));
  }

  // Make sure |Set| returns true when the bit was already set.
  for (int i = 3; i < 10000; i *= 2) {
    EXPECT_TRUE(bvec.Set(i));
  }
}

TEST(BitVectorTest, Clear) {
  BitVector bvec;
  for (int i = 3; i < 10000; i *= 2) {
    bvec.Set(i);
  }

  // Check that the bits were properly set.
  for (int i = 3; i < 10000; i *= 2) {
    EXPECT_TRUE(bvec.Get(i));
  }

  // Clear all of the bits except for bit 3.
  for (int i = 6; i < 10000; i *= 2) {
    bvec.Clear(i);
  }

  // Make sure bit 3 was not cleared.
  EXPECT_TRUE(bvec.Get(3));

  // Make sure all of the other bits that were set have been cleared.
  for (int i = 6; i < 10000; i *= 2) {
    EXPECT_FALSE(bvec.Get(i));
  }
}

TEST(BitVectorTest, ClearReturnValue) {
  BitVector bvec;
  for (int i = 3; i < 10000; i *= 2) {
    bvec.Set(i);
  }

  // Make sure |Clear| returns true if the bit was set.
  for (int i = 3; i < 10000; i *= 2) {
    EXPECT_TRUE(bvec.Clear(i));
  }

  // Make sure |Clear| returns false if the bit was not set.
  for (int i = 3; i < 10000; i *= 2) {
    EXPECT_FALSE(bvec.Clear(i));
  }
}

TEST(BitVectorTest, SimpleOrTest) {
  BitVector bvec1;
  bvec1.Set(3);
  bvec1.Set(4);

  BitVector bvec2;
  bvec2.Set(2);
  bvec2.Set(4);

  // Check that |bvec1| changed when doing the |Or| operation.
  EXPECT_TRUE(bvec1.Or(bvec2));

  // Check that the values are all correct.
  EXPECT_FALSE(bvec1.Get(0));
  EXPECT_FALSE(bvec1.Get(1));
  EXPECT_TRUE(bvec1.Get(2));
  EXPECT_TRUE(bvec1.Get(3));
  EXPECT_TRUE(bvec1.Get(4));
}

TEST(BitVectorTest, ResizingOrTest) {
  BitVector bvec1;
  bvec1.Set(3);
  bvec1.Set(4);

  BitVector bvec2;
  bvec2.Set(10000);

  // Similar to above except with a large value to test resizing.
  EXPECT_TRUE(bvec1.Or(bvec2));
  EXPECT_FALSE(bvec1.Get(0));
  EXPECT_FALSE(bvec1.Get(1));
  EXPECT_FALSE(bvec1.Get(2));
  EXPECT_TRUE(bvec1.Get(3));
  EXPECT_TRUE(bvec1.Get(10000));
}

TEST(BitVectorTest, SubsetOrTest) {
  BitVector bvec1;
  bvec1.Set(3);
  bvec1.Set(4);

  BitVector bvec2;
  bvec2.Set(3);

  // |Or| returns false if |bvec1| does not change.
  EXPECT_FALSE(bvec1.Or(bvec2));
}

}  // namespace
}  // namespace utils
}  // namespace spvtools
