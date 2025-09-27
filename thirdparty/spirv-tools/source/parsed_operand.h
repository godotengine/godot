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

#ifndef SOURCE_PARSED_OPERAND_H_
#define SOURCE_PARSED_OPERAND_H_

#include <ostream>

#include "spirv-tools/libspirv.h"

namespace spvtools {

// Emits the numeric literal representation of the given instruction operand
// to the stream.  The operand must be of numeric type.  If integral it may
// be up to 64 bits wide.  If floating point, then it must be 16, 32, or 64
// bits wide.
void EmitNumericLiteral(std::ostream* out, const spv_parsed_instruction_t& inst,
                        const spv_parsed_operand_t& operand);

}  // namespace spvtools

#endif  // SOURCE_PARSED_OPERAND_H_
