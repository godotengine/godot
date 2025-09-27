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

// Validates literal numbers.

#include "source/val/validate.h"

#include <cassert>

#include "source/diagnostic.h"
#include "source/opcode.h"
#include "source/val/instruction.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

// Returns true if the operand holds a literal number
bool IsLiteralNumber(const spv_parsed_operand_t& operand) {
  switch (operand.number_kind) {
    case SPV_NUMBER_SIGNED_INT:
    case SPV_NUMBER_UNSIGNED_INT:
    case SPV_NUMBER_FLOATING:
      return true;
    default:
      return false;
  }
}

// Verifies that the upper bits of the given upper |word| with given
// lower |width| are zero- or sign-extended when |signed_int| is true
bool VerifyUpperBits(uint32_t word, uint32_t width, bool signed_int) {
  assert(width < 32);
  assert(0 < width);
  const uint32_t upper_mask = 0xFFFFFFFFu << width;
  const uint32_t upper_bits = word & upper_mask;

  bool result = false;
  if (signed_int) {
    const uint32_t sign_bit = word & (1u << (width - 1));
    if (sign_bit) {
      result = upper_bits == upper_mask;
    } else {
      result = upper_bits == 0;
    }
  } else {
    result = upper_bits == 0;
  }
  return result;
}

}  // namespace

// Validates that literal numbers are represented according to the spec
spv_result_t LiteralsPass(ValidationState_t& _, const Instruction* inst) {
  // For every operand that is a literal number
  for (size_t i = 0; i < inst->operands().size(); i++) {
    const spv_parsed_operand_t& operand = inst->operand(i);
    if (!IsLiteralNumber(operand)) continue;

    // The upper bits are always in the last word (little-endian)
    int last_index = operand.offset + operand.num_words - 1;
    const uint32_t upper_word = inst->word(last_index);

    // TODO(jcaraban): is the |word size| defined in some header?
    const uint32_t word_size = 32;
    uint32_t bit_width = operand.number_bit_width;

    // Bit widths that are a multiple of the word size have no upper bits
    const auto remaining_value_bits = bit_width % word_size;
    if (remaining_value_bits == 0) continue;

    const bool signedness = operand.number_kind == SPV_NUMBER_SIGNED_INT;

    if (!VerifyUpperBits(upper_word, remaining_value_bits, signedness)) {
      return _.diag(SPV_ERROR_INVALID_VALUE, inst)
             << "The high-order bits of a literal number in instruction <id> "
             << inst->id() << " must be 0 for a floating-point type, "
             << "or 0 for an integer type with Signedness of 0, "
             << "or sign extended when Signedness is 1";
    }
  }
  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
