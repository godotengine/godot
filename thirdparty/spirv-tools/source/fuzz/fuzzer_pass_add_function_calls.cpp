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

#include "source/fuzz/fuzzer_pass_add_function_calls.h"

#include "source/fuzz/call_graph.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_add_global_variable.h"
#include "source/fuzz/transformation_add_local_variable.h"
#include "source/fuzz/transformation_function_call.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddFunctionCalls::FuzzerPassAddFunctionCalls(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassAddFunctionCalls::Apply() {
  ForEachInstructionWithInstructionDescriptor(
      [this](opt::Function* function, opt::BasicBlock* block,
             opt::BasicBlock::iterator inst_it,
             const protobufs::InstructionDescriptor& instruction_descriptor)
          -> void {
        // Check whether it is legitimate to insert a function call before the
        // instruction.
        if (!fuzzerutil::CanInsertOpcodeBeforeInstruction(
                spv::Op::OpFunctionCall, inst_it)) {
          return;
        }

        // Randomly decide whether to try inserting a function call here.
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfCallingFunction())) {
          return;
        }

        // Compute the module's call graph - we don't cache it since it may
        // change each time we apply a transformation.  If this proves to be
        // a bottleneck the call graph data structure could be made updatable.
        CallGraph call_graph(GetIRContext());

        // Gather all the non-entry point functions different from this
        // function.  It is important to ignore entry points as a function
        // cannot be an entry point and the target of an OpFunctionCall
        // instruction.  We ignore this function to avoid direct recursion.
        std::vector<opt::Function*> candidate_functions;
        for (auto& other_function : *GetIRContext()->module()) {
          if (&other_function != function &&
              !fuzzerutil::FunctionIsEntryPoint(GetIRContext(),
                                                other_function.result_id())) {
            candidate_functions.push_back(&other_function);
          }
        }

        // Choose a function to call, at random, by considering candidate
        // functions until a suitable one is found.
        opt::Function* chosen_function = nullptr;
        while (!candidate_functions.empty()) {
          opt::Function* candidate_function =
              GetFuzzerContext()->RemoveAtRandomIndex(&candidate_functions);
          if (!GetTransformationContext()->GetFactManager()->BlockIsDead(
                  block->id()) &&
              !GetTransformationContext()->GetFactManager()->FunctionIsLivesafe(
                  candidate_function->result_id())) {
            // Unless in a dead block, only livesafe functions can be invoked
            continue;
          }
          if (call_graph.GetIndirectCallees(candidate_function->result_id())
                  .count(function->result_id())) {
            // Calling this function could lead to indirect recursion
            continue;
          }
          chosen_function = candidate_function;
          break;
        }

        if (!chosen_function) {
          // No suitable function was found to call.  (This can happen, for
          // instance, if the current function is the only function in the
          // module.)
          return;
        }

        ApplyTransformation(TransformationFunctionCall(
            GetFuzzerContext()->GetFreshId(), chosen_function->result_id(),
            ChooseFunctionCallArguments(*chosen_function, function, block,
                                        inst_it),
            instruction_descriptor));
      });
}

std::vector<uint32_t> FuzzerPassAddFunctionCalls::ChooseFunctionCallArguments(
    const opt::Function& callee, opt::Function* caller_function,
    opt::BasicBlock* caller_block,
    const opt::BasicBlock::iterator& caller_inst_it) {
  auto available_pointers = FindAvailableInstructions(
      caller_function, caller_block, caller_inst_it,
      [this, caller_block](opt::IRContext* /*unused*/, opt::Instruction* inst) {
        if (inst->opcode() != spv::Op::OpVariable ||
            inst->opcode() != spv::Op::OpFunctionParameter) {
          // Function parameters and variables are the only
          // kinds of pointer that can be used as actual
          // parameters.
          return false;
        }

        return GetTransformationContext()->GetFactManager()->BlockIsDead(
                   caller_block->id()) ||
               GetTransformationContext()
                   ->GetFactManager()
                   ->PointeeValueIsIrrelevant(inst->result_id());
      });

  std::unordered_map<uint32_t, std::vector<uint32_t>> type_id_to_result_id;
  for (const auto* inst : available_pointers) {
    type_id_to_result_id[inst->type_id()].push_back(inst->result_id());
  }

  std::vector<uint32_t> result;
  for (const auto* param :
       fuzzerutil::GetParameters(GetIRContext(), callee.result_id())) {
    const auto* param_type =
        GetIRContext()->get_type_mgr()->GetType(param->type_id());
    assert(param_type && "Parameter has invalid type");

    if (!param_type->AsPointer()) {
      if (fuzzerutil::CanCreateConstant(GetIRContext(), param->type_id())) {
        // We mark the constant as irrelevant so that we can replace it with a
        // more interesting value later.
        result.push_back(FindOrCreateZeroConstant(param->type_id(), true));
      } else {
        result.push_back(FindOrCreateGlobalUndef(param->type_id()));
      }
      continue;
    }

    if (type_id_to_result_id.count(param->type_id())) {
      // Use an existing pointer if there are any.
      const auto& candidates = type_id_to_result_id[param->type_id()];
      result.push_back(candidates[GetFuzzerContext()->RandomIndex(candidates)]);
      continue;
    }

    // Make a new variable, at function or global scope depending on the storage
    // class of the pointer.

    // Get a fresh id for the new variable.
    uint32_t fresh_variable_id = GetFuzzerContext()->GetFreshId();

    // The id of this variable is what we pass as the parameter to
    // the call.
    result.push_back(fresh_variable_id);
    type_id_to_result_id[param->type_id()].push_back(fresh_variable_id);

    // Now bring the variable into existence.
    auto storage_class = param_type->AsPointer()->storage_class();
    auto pointee_type_id = fuzzerutil::GetPointeeTypeIdFromPointerType(
        GetIRContext(), param->type_id());
    if (storage_class == spv::StorageClass::Function) {
      // Add a new zero-initialized local variable to the current
      // function, noting that its pointee value is irrelevant.
      ApplyTransformation(TransformationAddLocalVariable(
          fresh_variable_id, param->type_id(), caller_function->result_id(),
          FindOrCreateZeroConstant(pointee_type_id, false), true));
    } else {
      assert((storage_class == spv::StorageClass::Private ||
              storage_class == spv::StorageClass::Workgroup) &&
             "Only Function, Private and Workgroup storage classes are "
             "supported at present.");
      // Add a new global variable to the module, zero-initializing it if
      // it has Private storage class, and noting that its pointee value is
      // irrelevant.
      ApplyTransformation(TransformationAddGlobalVariable(
          fresh_variable_id, param->type_id(), storage_class,
          storage_class == spv::StorageClass::Private
              ? FindOrCreateZeroConstant(pointee_type_id, false)
              : 0,
          true));
    }
  }

  return result;
}

}  // namespace fuzz
}  // namespace spvtools
