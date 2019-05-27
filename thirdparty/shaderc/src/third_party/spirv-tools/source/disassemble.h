// Copyright (c) 2018 Google Inc.
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

#ifndef SOURCE_DISASSEMBLE_H_
#define SOURCE_DISASSEMBLE_H_

#include <string>

#include "spirv-tools/libspirv.h"

namespace spvtools {

// Decodes the given SPIR-V instruction binary representation to its assembly
// text. The context is inferred from the provided module binary. The options
// parameter is a bit field of spv_binary_to_text_options_t. Decoded text will
// be stored into *text. Any error will be written into *diagnostic if
// diagnostic is non-null.
std::string spvInstructionBinaryToText(const spv_target_env env,
                                       const uint32_t* inst_binary,
                                       const size_t inst_word_count,
                                       const uint32_t* binary,
                                       const size_t word_count,
                                       const uint32_t options);

}  // namespace spvtools

#endif  // SOURCE_DISASSEMBLE_H_
