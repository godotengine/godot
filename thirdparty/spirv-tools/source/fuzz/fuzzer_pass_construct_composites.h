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

#ifndef SOURCE_FUZZ_FUZZER_PASS_CONSTRUCT_COMPOSITES_H_
#define SOURCE_FUZZ_FUZZER_PASS_CONSTRUCT_COMPOSITES_H_

#include <unordered_map>
#include <vector>

#include "source/fuzz/fuzzer_pass.h"

namespace spvtools {
namespace fuzz {

// A fuzzer pass for constructing composite objects from smaller objects.
class FuzzerPassConstructComposites : public FuzzerPass {
 public:
  FuzzerPassConstructComposites(
      opt::IRContext* ir_context, TransformationContext* transformation_context,
      FuzzerContext* fuzzer_context,
      protobufs::TransformationSequence* transformations,
      bool ignore_inapplicable_transformations);

  void Apply() override;

 private:
  // Used to map a type id to the ids of relevant instructions of the type.
  using TypeIdToInstructions =
      std::unordered_map<uint32_t, std::vector<uint32_t>>;

  // Requires that |array_type_instruction| has opcode OpTypeArray.
  // Attempts to find suitable instruction result ids from the values of
  // |type_id_to_available_instructions| that would allow a composite of type
  // |array_type_instruction| to be constructed.  Returns said ids if they can
  // be found and an empty vector otherwise.
  std::vector<uint32_t> FindComponentsToConstructArray(
      const opt::Instruction& array_type_instruction,
      const TypeIdToInstructions& type_id_to_available_instructions);

  // Similar to FindComponentsToConstructArray, but for matrices.
  std::vector<uint32_t> FindComponentsToConstructMatrix(
      const opt::Instruction& matrix_type_instruction,
      const TypeIdToInstructions& type_id_to_available_instructions);

  // Similar to FindComponentsToConstructArray, but for structs.
  std::vector<uint32_t> FindComponentsToConstructStruct(
      const opt::Instruction& struct_type_instruction,
      const TypeIdToInstructions& type_id_to_available_instructions);

  // Similar to FindComponentsToConstructArray, but for vectors.
  std::vector<uint32_t> FindComponentsToConstructVector(
      const opt::Instruction& vector_type_instruction,
      const TypeIdToInstructions& type_id_to_available_instructions);
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_PASS_CONSTRUCT_COMPOSITES_H_
