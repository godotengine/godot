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

#ifndef SOURCE_SPIRV_DEFINITION_H_
#define SOURCE_SPIRV_DEFINITION_H_

#include <cstdint>

#include "source/latest_version_spirv_header.h"

#define spvIsInBitfield(value, bitfield) ((value) == ((value)&bitfield))

typedef struct spv_header_t {
  uint32_t magic;
  uint32_t version;
  uint32_t generator;
  uint32_t bound;
  uint32_t schema;               // NOTE: Reserved
  const uint32_t* instructions;  // NOTE: Unfixed pointer to instruction stream
} spv_header_t;

#endif  // SOURCE_SPIRV_DEFINITION_H_
