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

#include "source/fuzz/fuzzer_pass_add_local_variables.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_add_local_variable.h"
#include "source/fuzz/transformation_add_type_pointer.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddLocalVariables::FuzzerPassAddLocalVariables(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassAddLocalVariables::Apply() {
  auto basic_type_ids_and_pointers =
      GetAvailableBasicTypesAndPointers(spv::StorageClass::Function);

  // These are the basic types that are available to this fuzzer pass.
  auto& basic_types = basic_type_ids_and_pointers.first;
  if (basic_types.empty()) {
    // The pass cannot do anything if there are no basic types.
    return;
  }

  // These are the pointers to those basic types that are *initially* available
  // to the fuzzer pass.  The fuzzer pass might add pointer types in cases where
  // none are available for a given basic type.
  auto& basic_type_to_pointers = basic_type_ids_and_pointers.second;

  // Consider every function in the module.
  for (auto& function : *GetIRContext()->module()) {
    // Probabilistically keep adding random variables to this function.
    while (GetFuzzerContext()->ChoosePercentage(
        GetFuzzerContext()->GetChanceOfAddingLocalVariable())) {
      // Choose a random basic type; the new variable's type will be a pointer
      // to this basic type.
      uint32_t basic_type =
          basic_types[GetFuzzerContext()->RandomIndex(basic_types)];
      uint32_t pointer_type;
      std::vector<uint32_t>& available_pointers_to_basic_type =
          basic_type_to_pointers.at(basic_type);
      // Determine whether there is at least one pointer to this basic type.
      if (available_pointers_to_basic_type.empty()) {
        // There is not.  Make one, to use here, and add it to the available
        // pointers for the basic type so that future variables can potentially
        // use it.
        pointer_type = GetFuzzerContext()->GetFreshId();
        ApplyTransformation(TransformationAddTypePointer(
            pointer_type, spv::StorageClass::Function, basic_type));
        available_pointers_to_basic_type.push_back(pointer_type);
      } else {
        // There is - grab one.
        pointer_type =
            available_pointers_to_basic_type[GetFuzzerContext()->RandomIndex(
                available_pointers_to_basic_type)];
      }
      ApplyTransformation(TransformationAddLocalVariable(
          GetFuzzerContext()->GetFreshId(), pointer_type, function.result_id(),
          FindOrCreateZeroConstant(basic_type, false), true));
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
