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

#include "source/extensions.h"

#include <cassert>
#include <sstream>
#include <string>

#include "source/binary.h"
#include "source/enum_string_mapping.h"

namespace spvtools {

std::string GetExtensionString(const spv_parsed_instruction_t* inst) {
  if (inst->opcode != static_cast<uint16_t>(spv::Op::OpExtension)) {
    return "ERROR_not_op_extension";
  }

  assert(inst->num_operands == 1);

  const auto& operand = inst->operands[0];
  assert(operand.type == SPV_OPERAND_TYPE_LITERAL_STRING);
  assert(inst->num_words > operand.offset);
  (void)operand; /* No unused variables in release builds. */

  return spvDecodeLiteralStringOperand(*inst, 0);
}

std::string ExtensionSetToString(const ExtensionSet& extensions) {
  std::stringstream ss;
  extensions.ForEach(
      [&ss](Extension ext) { ss << ExtensionToString(ext) << " "; });
  return ss.str();
}

}  // namespace spvtools
