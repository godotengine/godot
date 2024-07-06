// Copyright (c) 2019 Google LLC
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

#ifndef SOURCE_FUZZ_FUZZER_PASS_OBFUSCATE_CONSTANTS_H_
#define SOURCE_FUZZ_FUZZER_PASS_OBFUSCATE_CONSTANTS_H_

#include <vector>

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {

// A fuzzer pass for turning uses of constants into more complex forms.
// Examples include replacing 'true' with '42 < 52', and replacing '42' with
// 'a.b.c' if 'a.b.c' is known to hold the value '42'.
class FuzzerPassObfuscateConstants : public FuzzerPass {
 public:
  FuzzerPassObfuscateConstants(
      opt::IRContext* ir_context, TransformationContext* transformation_context,
      FuzzerContext* fuzzer_context,
      protobufs::TransformationSequence* transformations,
      bool ignore_inapplicable_transformations);

  void Apply() override;

 private:
  // Applies 0 or more transformations to potentially obfuscate the constant
  // use represented by |constant_use|.  The |depth| parameter controls how
  // deeply obfuscation can recurse.
  void ObfuscateConstant(uint32_t depth,
                         const protobufs::IdUseDescriptor& constant_use);

  // This method will try to turn |constant_use|, required to be a use of a
  // boolean constant, into a binary expression on scalar constants, which may
  // themselves be recursively obfuscated.
  void ObfuscateBoolConstant(uint32_t depth,
                             const protobufs::IdUseDescriptor& constant_use);

  // This method will try to turn |constant_use|, required to be a use of a
  // scalar constant, into the value loaded from a uniform known to have the
  // same value as the constant (if one exists).
  void ObfuscateScalarConstant(uint32_t depth,
                               const protobufs::IdUseDescriptor& constant_use);

  // Applies a transformation to replace the boolean constant usage represented
  // by |bool_constant_use| with a binary expression involving
  // |float_constant_id_1| and |float_constant_id_2|, which must not be equal
  // to one another.  Possibly further obfuscates the uses of these float
  // constants.  The |depth| parameter controls how deeply obfuscation can
  // recurse.
  void ObfuscateBoolConstantViaFloatConstantPair(
      uint32_t depth, const protobufs::IdUseDescriptor& bool_constant_use,
      uint32_t float_constant_id_1, uint32_t float_constant_id_2);

  // Similar to the above, but for signed int constants.
  void ObfuscateBoolConstantViaSignedIntConstantPair(
      uint32_t depth, const protobufs::IdUseDescriptor& bool_constant_use,
      uint32_t signed_int_constant_id_1, uint32_t signed_int_constant_id_2);

  // Similar to the above, but for unsigned int constants.
  void ObfuscateBoolConstantViaUnsignedIntConstantPair(
      uint32_t depth, const protobufs::IdUseDescriptor& bool_constant_use,
      uint32_t unsigned_int_constant_id_1, uint32_t unsigned_int_constant_id_2);

  // A helper method to capture the common parts of the above methods.
  // The method is used to obfuscate the boolean constant usage represented by
  // |bool_constant_use| by replacing it with '|constant_id_1| OP
  // |constant_id_2|', where 'OP' is chosen from either |greater_than_opcodes|
  // or |less_than_opcodes|.
  //
  // The two constant ids must not represent the same value, and thus
  // |greater_than_opcodes| may include 'greater than or equal' opcodes
  // (similar for |less_than_opcodes|).
  void ObfuscateBoolConstantViaConstantPair(
      uint32_t depth, const protobufs::IdUseDescriptor& bool_constant_use,
      const std::vector<spv::Op>& greater_than_opcodes,
      const std::vector<spv::Op>& less_than_opcodes, uint32_t constant_id_1,
      uint32_t constant_id_2, bool first_constant_is_larger);

  // A helper method to determine whether input operand |in_operand_index| of
  // |inst| is the id of a constant, and add an id use descriptor to
  // |candidate_constant_uses| if so.  The other parameters are used for id use
  // descriptor construction.
  void MaybeAddConstantIdUse(
      const opt::Instruction& inst, uint32_t in_operand_index,
      uint32_t base_instruction_result_id,
      const std::map<spv::Op, uint32_t>& skipped_opcode_count,
      std::vector<protobufs::IdUseDescriptor>* constant_uses);

  // Returns a vector of unique words that denote constants. Every such constant
  // is used in |FactConstantUniform| and has type with id equal to |type_id|.
  std::vector<std::vector<uint32_t>> GetConstantWordsFromUniformsForType(
      uint32_t type_id);
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_PASS_OBFUSCATE_CONSTANTS_H_
