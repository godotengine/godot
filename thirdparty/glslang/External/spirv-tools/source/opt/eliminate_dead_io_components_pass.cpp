// Copyright (c) 2022 The Khronos Group Inc.
// Copyright (c) 2022 LunarG Inc.
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

#include "source/opt/eliminate_dead_io_components_pass.h"

#include <vector>

#include "source/opt/instruction.h"
#include "source/opt/ir_context.h"
#include "source/util/bit_vector.h"

namespace spvtools {
namespace opt {
namespace {
constexpr uint32_t kAccessChainBaseInIdx = 0;
constexpr uint32_t kAccessChainIndex0InIdx = 1;
constexpr uint32_t kAccessChainIndex1InIdx = 2;
constexpr uint32_t kConstantValueInIdx = 0;
}  // namespace

Pass::Status EliminateDeadIOComponentsPass::Process() {
  // Only process input and output variables
  if (elim_sclass_ != spv::StorageClass::Input &&
      elim_sclass_ != spv::StorageClass::Output) {
    if (consumer()) {
      std::string message =
          "EliminateDeadIOComponentsPass only valid for input and output "
          "variables.";
      consumer()(SPV_MSG_ERROR, 0, {0, 0, 0}, message.c_str());
    }
    return Status::Failure;
  }
  // If safe mode, only process Input variables in vertex shader
  const auto stage = context()->GetStage();
  if (safe_mode_ && !(stage == spv::ExecutionModel::Vertex &&
                      elim_sclass_ == spv::StorageClass::Input))
    return Status::SuccessWithoutChange;
  // Current functionality assumes shader capability.
  if (!context()->get_feature_mgr()->HasCapability(spv::Capability::Shader))
    return Status::SuccessWithoutChange;
  // Current functionality assumes vert, frag, tesc, tese or geom shader.
  // TODO(issue #4988): Add GLCompute.
  if (stage != spv::ExecutionModel::Vertex &&
      stage != spv::ExecutionModel::Fragment &&
      stage != spv::ExecutionModel::TessellationControl &&
      stage != spv::ExecutionModel::TessellationEvaluation &&
      stage != spv::ExecutionModel::Geometry)
    return Status::SuccessWithoutChange;
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  bool modified = false;
  std::vector<Instruction*> vars_to_move;
  for (auto& var : context()->types_values()) {
    if (var.opcode() != spv::Op::OpVariable) {
      continue;
    }
    analysis::Type* var_type = type_mgr->GetType(var.type_id());
    analysis::Pointer* ptr_type = var_type->AsPointer();
    if (ptr_type == nullptr) {
      continue;
    }
    const auto sclass = ptr_type->storage_class();
    if (sclass != elim_sclass_) {
      continue;
    }
    // For tesc, or input variables in tese or geom shaders,
    // there is a outer per-vertex-array that must be ignored
    // for the purposes of this analysis/optimization. Do the
    // analysis on the inner type in these cases.
    bool skip_first_index = false;
    auto core_type = ptr_type->pointee_type();
    if (stage == spv::ExecutionModel::TessellationControl ||
        (sclass == spv::StorageClass::Input &&
         (stage == spv::ExecutionModel::TessellationEvaluation ||
          stage == spv::ExecutionModel::Geometry))) {
      auto arr_type = core_type->AsArray();
      if (!arr_type) continue;
      core_type = arr_type->element_type();
      skip_first_index = true;
    }
    const analysis::Array* arr_type = core_type->AsArray();
    if (arr_type != nullptr) {
      // Only process array if input of vertex shader, or output of
      // fragment shader. Otherwise, if one shader has a runtime index and the
      // other does not, interface incompatibility can occur.
      if (!((sclass == spv::StorageClass::Input &&
             stage == spv::ExecutionModel::Vertex) ||
            (sclass == spv::StorageClass::Output &&
             stage == spv::ExecutionModel::Fragment)))
        continue;
      unsigned arr_len_id = arr_type->LengthId();
      Instruction* arr_len_inst = def_use_mgr->GetDef(arr_len_id);
      if (arr_len_inst->opcode() != spv::Op::OpConstant) {
        continue;
      }
      // SPIR-V requires array size is >= 1, so this works for signed or
      // unsigned size.
      unsigned original_max =
          arr_len_inst->GetSingleWordInOperand(kConstantValueInIdx) - 1;
      unsigned max_idx = FindMaxIndex(var, original_max);
      if (max_idx != original_max) {
        ChangeArrayLength(var, max_idx + 1);
        vars_to_move.push_back(&var);
        modified = true;
      }
      continue;
    }
    const analysis::Struct* struct_type = core_type->AsStruct();
    if (struct_type == nullptr) continue;
    const auto elt_types = struct_type->element_types();
    unsigned original_max = static_cast<unsigned>(elt_types.size()) - 1;
    unsigned max_idx = FindMaxIndex(var, original_max, skip_first_index);
    if (max_idx != original_max) {
      ChangeIOVarStructLength(var, max_idx + 1);
      vars_to_move.push_back(&var);
      modified = true;
    }
  }

  // Move changed vars after their new type instruction to preserve backward
  // referencing.
  for (auto var : vars_to_move) {
    auto type_id = var->type_id();
    auto type_inst = def_use_mgr->GetDef(type_id);
    var->RemoveFromList();
    var->InsertAfter(type_inst);
  }

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

unsigned EliminateDeadIOComponentsPass::FindMaxIndex(
    const Instruction& var, const unsigned original_max,
    const bool skip_first_index) {
  unsigned max = 0;
  bool seen_non_const_ac = false;
  assert(var.opcode() == spv::Op::OpVariable && "must be variable");
  context()->get_def_use_mgr()->WhileEachUser(
      var.result_id(), [&max, &seen_non_const_ac, var, skip_first_index,
                        this](Instruction* use) {
        auto use_opcode = use->opcode();
        if (use_opcode == spv::Op::OpLoad || use_opcode == spv::Op::OpStore ||
            use_opcode == spv::Op::OpCopyMemory ||
            use_opcode == spv::Op::OpCopyMemorySized ||
            use_opcode == spv::Op::OpCopyObject) {
          seen_non_const_ac = true;
          return false;
        }
        if (use->opcode() != spv::Op::OpAccessChain &&
            use->opcode() != spv::Op::OpInBoundsAccessChain) {
          return true;
        }
        // OpAccessChain with no indices currently not optimized
        if (use->NumInOperands() == 1 ||
            (skip_first_index && use->NumInOperands() == 2)) {
          seen_non_const_ac = true;
          return false;
        }
        const unsigned base_id =
            use->GetSingleWordInOperand(kAccessChainBaseInIdx);
        USE_ASSERT(base_id == var.result_id() && "unexpected base");
        const unsigned in_idx = skip_first_index ? kAccessChainIndex1InIdx
                                                 : kAccessChainIndex0InIdx;
        const unsigned idx_id = use->GetSingleWordInOperand(in_idx);
        Instruction* idx_inst = context()->get_def_use_mgr()->GetDef(idx_id);
        if (idx_inst->opcode() != spv::Op::OpConstant) {
          seen_non_const_ac = true;
          return false;
        }
        unsigned value = idx_inst->GetSingleWordInOperand(kConstantValueInIdx);
        if (value > max) max = value;
        return true;
      });
  return seen_non_const_ac ? original_max : max;
}

void EliminateDeadIOComponentsPass::ChangeArrayLength(Instruction& arr_var,
                                                      unsigned length) {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::ConstantManager* const_mgr = context()->get_constant_mgr();
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  analysis::Pointer* ptr_type =
      type_mgr->GetType(arr_var.type_id())->AsPointer();
  const analysis::Array* arr_ty = ptr_type->pointee_type()->AsArray();
  assert(arr_ty && "expecting array type");
  uint32_t length_id = const_mgr->GetUIntConstId(length);
  analysis::Array new_arr_ty(arr_ty->element_type(),
                             arr_ty->GetConstantLengthInfo(length_id, length));
  analysis::Type* reg_new_arr_ty = type_mgr->GetRegisteredType(&new_arr_ty);
  analysis::Pointer new_ptr_ty(reg_new_arr_ty, ptr_type->storage_class());
  analysis::Type* reg_new_ptr_ty = type_mgr->GetRegisteredType(&new_ptr_ty);
  uint32_t new_ptr_ty_id = type_mgr->GetTypeInstruction(reg_new_ptr_ty);
  arr_var.SetResultType(new_ptr_ty_id);
  def_use_mgr->AnalyzeInstUse(&arr_var);
}

void EliminateDeadIOComponentsPass::ChangeIOVarStructLength(Instruction& io_var,
                                                            unsigned length) {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::Pointer* ptr_type =
      type_mgr->GetType(io_var.type_id())->AsPointer();
  auto core_type = ptr_type->pointee_type();
  // Check for per-vertex-array of struct from tesc, tese and geom and grab
  // embedded struct type.
  const auto arr_type = core_type->AsArray();
  if (arr_type) core_type = arr_type->element_type();
  const analysis::Struct* struct_ty = core_type->AsStruct();
  assert(struct_ty && "expecting struct type");
  const auto orig_elt_types = struct_ty->element_types();
  std::vector<const analysis::Type*> new_elt_types;
  for (unsigned u = 0; u < length; ++u)
    new_elt_types.push_back(orig_elt_types[u]);
  analysis::Struct new_struct_ty(new_elt_types);
  uint32_t old_struct_ty_id = type_mgr->GetTypeInstruction(struct_ty);
  std::vector<Instruction*> decorations =
      context()->get_decoration_mgr()->GetDecorationsFor(old_struct_ty_id,
                                                         true);
  for (auto dec : decorations) {
    if (dec->opcode() == spv::Op::OpMemberDecorate) {
      uint32_t midx = dec->GetSingleWordInOperand(1);
      if (midx >= length) continue;
    }
    type_mgr->AttachDecoration(*dec, &new_struct_ty);
  }
  // Clone name instructions for new struct type
  analysis::Type* reg_new_str_ty = type_mgr->GetRegisteredType(&new_struct_ty);
  uint32_t new_struct_ty_id = type_mgr->GetTypeInstruction(reg_new_str_ty);
  context()->CloneNames(old_struct_ty_id, new_struct_ty_id, length);
  // Attach new type to var
  analysis::Type* reg_new_var_ty = reg_new_str_ty;
  if (arr_type) {
    analysis::Array new_arr_ty(reg_new_var_ty, arr_type->length_info());
    reg_new_var_ty = type_mgr->GetRegisteredType(&new_arr_ty);
  }
  analysis::Pointer new_ptr_ty(reg_new_var_ty, elim_sclass_);
  analysis::Type* reg_new_ptr_ty = type_mgr->GetRegisteredType(&new_ptr_ty);
  uint32_t new_ptr_ty_id = type_mgr->GetTypeInstruction(reg_new_ptr_ty);
  io_var.SetResultType(new_ptr_ty_id);
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  def_use_mgr->AnalyzeInstUse(&io_var);
}

}  // namespace opt
}  // namespace spvtools
