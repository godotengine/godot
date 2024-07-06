// Copyright (c) 2020 Google LLC
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

#ifndef SOURCE_FUZZ_FUZZER_PASS_ADD_EQUATION_INSTRUCTIONS_H_
#define SOURCE_FUZZ_FUZZER_PASS_ADD_EQUATION_INSTRUCTIONS_H_

#include <vector>

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {

// Fuzzer pass that sprinkles instructions through the module that define
// equations using various arithmetic and logical operators.
class FuzzerPassAddEquationInstructions : public FuzzerPass {
 public:
  FuzzerPassAddEquationInstructions(
      opt::IRContext* ir_context, TransformationContext* transformation_context,
      FuzzerContext* fuzzer_context,
      protobufs::TransformationSequence* transformations,
      bool ignore_inapplicable_transformations);

  void Apply() override;

 private:
  // Yields those instructions in |instructions| that have integer scalar or
  // vector result type.
  std::vector<opt::Instruction*> GetIntegerInstructions(
      const std::vector<opt::Instruction*>& instructions) const;

  // Returns only instructions, that have either a scalar floating-point or a
  // vector type.
  std::vector<opt::Instruction*> GetFloatInstructions(
      const std::vector<opt::Instruction*>& instructions) const;

  // Yields those instructions in |instructions| that have boolean scalar or
  // vector result type.
  std::vector<opt::Instruction*> GetBooleanInstructions(
      const std::vector<opt::Instruction*>& instructions) const;

  // Yields those instructions in |instructions| that have a scalar numerical or
  // a vector of numerical components type. Only 16, 32 and 64-bit numericals
  // are supported if both OpTypeInt and OpTypeFloat instructions can be created
  // with the specified width (e.g. for 16-bit types both Float16 and Int16
  // capabilities must be present).
  std::vector<opt::Instruction*> GetNumericalInstructions(
      const std::vector<opt::Instruction*>& instructions) const;

  // Requires that |instructions| are scalars or vectors of some type.  Returns
  // only those instructions whose width is |width|. If |width| is 1 this means
  // the scalars.
  std::vector<opt::Instruction*> RestrictToVectorWidth(
      const std::vector<opt::Instruction*>& instructions,
      uint32_t vector_width) const;

  // Requires that |instructions| are integer or float scalars or vectors.
  // Returns only those instructions for which the bit-width of the underlying
  // integer or floating-point type is |bit_width|.
  std::vector<opt::Instruction*> RestrictToElementBitWidth(
      const std::vector<opt::Instruction*>& instructions,
      uint32_t bit_width) const;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_PASS_ADD_EQUATION_INSTRUCTIONS_H_
