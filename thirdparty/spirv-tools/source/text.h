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

#ifndef SOURCE_TEXT_H_
#define SOURCE_TEXT_H_

#include <string>

#include "source/operand.h"
#include "source/spirv_constant.h"
#include "spirv-tools/libspirv.h"

typedef enum spv_literal_type_t {
  SPV_LITERAL_TYPE_INT_32,
  SPV_LITERAL_TYPE_INT_64,
  SPV_LITERAL_TYPE_UINT_32,
  SPV_LITERAL_TYPE_UINT_64,
  SPV_LITERAL_TYPE_FLOAT_32,
  SPV_LITERAL_TYPE_FLOAT_64,
  SPV_LITERAL_TYPE_STRING,
  SPV_FORCE_32_BIT_ENUM(spv_literal_type_t)
} spv_literal_type_t;

typedef struct spv_literal_t {
  spv_literal_type_t type;
  union value_t {
    int32_t i32;
    int64_t i64;
    uint32_t u32;
    uint64_t u64;
    float f;
    double d;
  } value;
  std::string str;  // Special field for literal string.
} spv_literal_t;

// Converts the given text string to a number/string literal and writes the
// result to *literal. String literals must be surrounded by double-quotes ("),
// which are then stripped.
spv_result_t spvTextToLiteral(const char* text, spv_literal_t* literal);

#endif  // SOURCE_TEXT_H_
