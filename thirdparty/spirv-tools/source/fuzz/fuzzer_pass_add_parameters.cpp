// Copyright (c) 2020 Vasyl Teliman
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

#include "source/fuzz/fuzzer_pass_add_parameters.h"

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_add_parameter.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddParameters::FuzzerPassAddParameters(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassAddParameters::Apply() {
  // Compute type candidates for the new parameter.
  std::vector<uint32_t> type_candidates;
  for (const auto& type_inst : GetIRContext()->module()->GetTypes()) {
    if (TransformationAddParameter::IsParameterTypeSupported(
            GetIRContext(), type_inst->result_id())) {
      type_candidates.push_back(type_inst->result_id());
    }
  }

  if (type_candidates.empty()) {
    // The module contains no suitable types to use in new parameters.
    return;
  }

  // Iterate over all functions in the module.
  for (const auto& function : *GetIRContext()->module()) {
    // Skip all entry-point functions - we don't want to change those.
    if (fuzzerutil::FunctionIsEntryPoint(GetIRContext(),
                                         function.result_id())) {
      continue;
    }

    if (GetNumberOfParameters(function) >=
        GetFuzzerContext()->GetMaximumNumberOfFunctionParameters()) {
      continue;
    }

    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()->GetChanceOfAddingParameters())) {
      continue;
    }

    auto num_new_parameters =
        GetFuzzerContext()->GetRandomNumberOfNewParameters(
            GetNumberOfParameters(function));

    for (uint32_t i = 0; i < num_new_parameters; ++i) {
      auto current_type_id =
          type_candidates[GetFuzzerContext()->RandomIndex(type_candidates)];
      auto current_type =
          GetIRContext()->get_type_mgr()->GetType(current_type_id);
      std::map<uint32_t, uint32_t> call_parameter_ids;

      // Consider the case when a pointer type was selected.
      if (current_type->kind() == opt::analysis::Type::kPointer) {
        auto storage_class = fuzzerutil::GetStorageClassFromPointerType(
            GetIRContext(), current_type_id);
        switch (storage_class) {
          case spv::StorageClass::Function: {
            // In every caller find or create a local variable that has the
            // selected type.
            for (auto* instr :
                 fuzzerutil::GetCallers(GetIRContext(), function.result_id())) {
              auto block = GetIRContext()->get_instr_block(instr);
              auto function_id = block->GetParent()->result_id();
              uint32_t variable_id =
                  FindOrCreateLocalVariable(current_type_id, function_id, true);
              call_parameter_ids[instr->result_id()] = variable_id;
            }
          } break;
          case spv::StorageClass::Private:
          case spv::StorageClass::Workgroup: {
            // If there exists at least one caller, find or create a global
            // variable that has the selected type.
            std::vector<opt::Instruction*> callers =
                fuzzerutil::GetCallers(GetIRContext(), function.result_id());
            if (!callers.empty()) {
              uint32_t variable_id =
                  FindOrCreateGlobalVariable(current_type_id, true);
              for (auto* instr : callers) {
                call_parameter_ids[instr->result_id()] = variable_id;
              }
            }
          } break;
          default:
            break;
        }
      } else {
        // If there exists at least one caller, find or create a zero constant
        // that has the selected type.
        std::vector<opt::Instruction*> callers =
            fuzzerutil::GetCallers(GetIRContext(), function.result_id());
        if (!callers.empty()) {
          uint32_t constant_id =
              FindOrCreateZeroConstant(current_type_id, true);
          for (auto* instr :
               fuzzerutil::GetCallers(GetIRContext(), function.result_id())) {
            call_parameter_ids[instr->result_id()] = constant_id;
          }
        }
      }

      ApplyTransformation(TransformationAddParameter(
          function.result_id(), GetFuzzerContext()->GetFreshId(),
          current_type_id, std::move(call_parameter_ids),
          GetFuzzerContext()->GetFreshId()));
    }
  }
}

uint32_t FuzzerPassAddParameters::GetNumberOfParameters(
    const opt::Function& function) const {
  const auto* type = GetIRContext()->get_type_mgr()->GetType(
      function.DefInst().GetSingleWordInOperand(1));
  assert(type && type->AsFunction());

  return static_cast<uint32_t>(type->AsFunction()->param_types().size());
}

}  // namespace fuzz
}  // namespace spvtools
