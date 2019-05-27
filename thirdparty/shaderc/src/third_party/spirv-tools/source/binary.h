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

#ifndef SOURCE_BINARY_H_
#define SOURCE_BINARY_H_

#include "source/spirv_definition.h"
#include "spirv-tools/libspirv.h"

// Functions

// Grabs the header from the SPIR-V module given in the binary parameter. The
// endian parameter specifies the endianness of the binary module. On success,
// returns SPV_SUCCESS and writes the parsed header into *header.
spv_result_t spvBinaryHeaderGet(const spv_const_binary binary,
                                const spv_endianness_t endian,
                                spv_header_t* header);

// Returns the number of non-null characters in str before the first null
// character, or strsz if there is no null character.  Examines at most the
// first strsz characters in str.  Returns 0 if str is nullptr.  This is a
// replacement for C11's strnlen_s which might not exist in all environments.
size_t spv_strnlen_s(const char* str, size_t strsz);

#endif  // SOURCE_BINARY_H_
