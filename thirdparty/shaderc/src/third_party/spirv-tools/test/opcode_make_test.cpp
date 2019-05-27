// Copyright (c) 2015-2016 The Khronos Group Inc.
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

#include "test/unit_spirv.h"

namespace spvtools {
namespace {

// A sampling of word counts.  Covers extreme points well, and all bit
// positions, and some combinations of bit positions.
const uint16_t kSampleWordCounts[] = {
    0,   1,   2,   3,    4,    8,    16,   32,    64,    127,    128,
    256, 511, 512, 1024, 2048, 4096, 8192, 16384, 32768, 0xfffe, 0xffff};

// A sampling of opcode values.  Covers the lower values well, a few samples
// around the number of core instructions (as of this writing), and some
// higher values.
const uint16_t kSampleOpcodes[] = {0,   1,   2,    3,      4,     100,
                                   300, 305, 1023, 0xfffe, 0xffff};

TEST(OpcodeMake, Samples) {
  for (auto wordCount : kSampleWordCounts) {
    for (auto opcode : kSampleOpcodes) {
      uint32_t word = 0;
      word |= uint32_t(opcode);
      word |= uint32_t(wordCount) << 16;
      EXPECT_EQ(word, spvOpcodeMake(wordCount, SpvOp(opcode)));
    }
  }
}

}  // namespace
}  // namespace spvtools
