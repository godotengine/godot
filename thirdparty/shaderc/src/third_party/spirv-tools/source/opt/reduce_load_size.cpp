// Copyright (c) 2018 Google LLC
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

#include "source/opt/reduce_load_size.h"

#include <set>
#include <vector>

#include "source/opt/instruction.h"
#include "source/opt/ir_builder.h"
#include "source/opt/ir_context.h"
#include "source/util/bit_vector.h"

namespace {

const uint32_t kExtractCompositeIdInIdx = 0;
const uint32_t kVariableStorageClassInIdx = 0;
const uint32_t kLoadPointerInIdx = 0;
const double kThreshold = 0.9;

}  // namespace

namespace spvtools {
namespace opt {

Pass::Status ReduceLoadSize::Process() {
  bool modified = false;

  for (auto& func : *get_module()) {
    func.ForEachInst([&modified, this](Instruction* inst) {
      if (inst->opcode() == SpvOpCompositeExtract) {
        if (ShouldReplaceExtract(inst)) {
          modified |= ReplaceExtract(inst);
        }
      }
    });
  }

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

bool ReduceLoadSize::ReplaceExtract(Instruction* inst) {
  assert(inst->opcode() == SpvOpCompositeExtract &&
         "Wrong opcode.  Should be OpCompositeExtract.");
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::ConstantManager* const_mgr = context()->get_constant_mgr();

  uint32_t composite_id =
      inst->GetSingleWordInOperand(kExtractCompositeIdInIdx);
  Instruction* composite_inst = def_use_mgr->GetDef(composite_id);

  if (composite_inst->opcode() != SpvOpLoad) {
    return false;
  }

  analysis::Type* composite_type = type_mgr->GetType(composite_inst->type_id());
  if (composite_type->kind() == analysis::Type::kVector ||
      composite_type->kind() == analysis::Type::kMatrix) {
    return false;
  }

  Instruction* var = composite_inst->GetBaseAddress();
  if (var == nullptr || var->opcode() != SpvOpVariable) {
    return false;
  }

  SpvStorageClass storage_class = static_cast<SpvStorageClass>(
      var->GetSingleWordInOperand(kVariableStorageClassInIdx));
  switch (storage_class) {
    case SpvStorageClassUniform:
    case SpvStorageClassUniformConstant:
    case SpvStorageClassInput:
      break;
    default:
      return false;
  }

  // Create a new access chain and load just after the old load.
  // We cannot create the new access chain load in the position of the extract
  // because the storage may have been written to in between.
  InstructionBuilder ir_builder(
      inst->context(), composite_inst,
      IRContext::kAnalysisInstrToBlockMapping | IRContext::kAnalysisDefUse);

  uint32_t pointer_to_result_type_id =
      type_mgr->FindPointerToType(inst->type_id(), storage_class);
  assert(pointer_to_result_type_id != 0 &&
         "We did not find the pointer type that we need.");

  analysis::Integer int_type(32, false);
  const analysis::Type* uint32_type = type_mgr->GetRegisteredType(&int_type);
  std::vector<uint32_t> ids;
  for (uint32_t i = 1; i < inst->NumInOperands(); ++i) {
    uint32_t index = inst->GetSingleWordInOperand(i);
    const analysis::Constant* index_const =
        const_mgr->GetConstant(uint32_type, {index});
    ids.push_back(const_mgr->GetDefiningInstruction(index_const)->result_id());
  }

  Instruction* new_access_chain = ir_builder.AddAccessChain(
      pointer_to_result_type_id,
      composite_inst->GetSingleWordInOperand(kLoadPointerInIdx), ids);
  Instruction* new_laod =
      ir_builder.AddLoad(inst->type_id(), new_access_chain->result_id());

  context()->ReplaceAllUsesWith(inst->result_id(), new_laod->result_id());
  context()->KillInst(inst);
  return true;
}

bool ReduceLoadSize::ShouldReplaceExtract(Instruction* inst) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  Instruction* op_inst = def_use_mgr->GetDef(
      inst->GetSingleWordInOperand(kExtractCompositeIdInIdx));

  if (op_inst->opcode() != SpvOpLoad) {
    return false;
  }

  auto cached_result = should_replace_cache_.find(op_inst->result_id());
  if (cached_result != should_replace_cache_.end()) {
    return cached_result->second;
  }

  bool all_elements_used = false;
  std::set<uint32_t> elements_used;

  all_elements_used =
      !def_use_mgr->WhileEachUser(op_inst, [&elements_used](Instruction* use) {
        if (use->opcode() != SpvOpCompositeExtract ||
            use->NumInOperands() == 1) {
          return false;
        }
        elements_used.insert(use->GetSingleWordInOperand(1));
        return true;
      });

  bool should_replace = false;
  if (all_elements_used) {
    should_replace = false;
  } else {
    analysis::ConstantManager* const_mgr = context()->get_constant_mgr();
    analysis::TypeManager* type_mgr = context()->get_type_mgr();
    analysis::Type* load_type = type_mgr->GetType(op_inst->type_id());
    uint32_t total_size = 1;
    switch (load_type->kind()) {
      case analysis::Type::kArray: {
        const analysis::Constant* size_const =
            const_mgr->FindDeclaredConstant(load_type->AsArray()->LengthId());
        assert(size_const->AsIntConstant());
        total_size = size_const->GetU32();
      } break;
      case analysis::Type::kStruct:
        total_size = static_cast<uint32_t>(
            load_type->AsStruct()->element_types().size());
        break;
      default:
        break;
    }
    double percent_used = static_cast<double>(elements_used.size()) /
                          static_cast<double>(total_size);
    should_replace = (percent_used < kThreshold);
  }

  should_replace_cache_[op_inst->result_id()] = should_replace;
  return should_replace;
}

}  // namespace opt
}  // namespace spvtools
