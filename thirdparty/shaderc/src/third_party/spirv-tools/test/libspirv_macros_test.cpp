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

TEST(Macros, BitShiftInnerParens) { ASSERT_EQ(65536, SPV_BIT(2 << 3)); }

TEST(Macros, BitShiftOuterParens) { ASSERT_EQ(15, SPV_BIT(4) - 1); }

}  // namespace
}  // namespace spvtools
