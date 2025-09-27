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

#include "source/opt/eliminate_dead_output_stores_pass.h"

#include "source/opt/instruction.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {
namespace {
constexpr uint32_t kDecorationLocationInIdx = 2;
constexpr uint32_t kOpDecorateMemberMemberInIdx = 1;
constexpr uint32_t kOpDecorateBuiltInLiteralInIdx = 2;
constexpr uint32_t kOpDecorateMemberBuiltInLiteralInIdx = 3;
constexpr uint32_t kOpAccessChainIdx0InIdx = 1;
constexpr uint32_t kOpConstantValueInIdx = 0;
}  // namespace

Pass::Status EliminateDeadOutputStoresPass::Process() {
  // Current functionality assumes shader capability
  if (!context()->get_feature_mgr()->HasCapability(spv::Capability::Shader))
    return Status::SuccessWithoutChange;
  Pass::Status status = DoDeadOutputStoreElimination();
  return status;
}

void EliminateDeadOutputStoresPass::InitializeElimination() {
  kill_list_.clear();
}

bool EliminateDeadOutputStoresPass::IsLiveBuiltin(uint32_t bi) {
  return live_builtins_->find(bi) != live_builtins_->end();
}

bool EliminateDeadOutputStoresPass::AnyLocsAreLive(uint32_t start,
                                                   uint32_t count) {
  auto finish = start + count;
  for (uint32_t u = start; u < finish; ++u) {
    if (live_locs_->find(u) != live_locs_->end()) return true;
  }
  return false;
}

void EliminateDeadOutputStoresPass::KillAllStoresOfRef(Instruction* ref) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  if (ref->opcode() == spv::Op::OpStore) {
    kill_list_.push_back(ref);
    return;
  }
  assert((ref->opcode() == spv::Op::OpAccessChain ||
          ref->opcode() == spv::Op::OpInBoundsAccessChain) &&
         "unexpected use of output variable");
  def_use_mgr->ForEachUser(ref, [this](Instruction* user) {
    if (user->opcode() == spv::Op::OpStore) kill_list_.push_back(user);
  });
}

void EliminateDeadOutputStoresPass::KillAllDeadStoresOfLocRef(
    Instruction* ref, Instruction* var) {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::DecorationManager* deco_mgr = context()->get_decoration_mgr();
  analysis::LivenessManager* live_mgr = context()->get_liveness_mgr();
  // Find variable location if present.
  uint32_t start_loc = 0;
  auto var_id = var->result_id();
  bool no_loc = deco_mgr->WhileEachDecoration(
      var_id, uint32_t(spv::Decoration::Location),
      [&start_loc](const Instruction& deco) {
        assert(deco.opcode() == spv::Op::OpDecorate && "unexpected decoration");
        start_loc = deco.GetSingleWordInOperand(kDecorationLocationInIdx);
        return false;
      });
  // Find patch decoration if present
  bool is_patch = !deco_mgr->WhileEachDecoration(
      var_id, uint32_t(spv::Decoration::Patch), [](const Instruction& deco) {
        if (deco.opcode() != spv::Op::OpDecorate)
          assert(false && "unexpected decoration");
        return false;
      });
  // Compute offset and final type of reference. If no location found
  // or any stored locations are live, return without removing stores.
  auto ptr_type = type_mgr->GetType(var->type_id())->AsPointer();
  assert(ptr_type && "unexpected var type");
  auto var_type = ptr_type->pointee_type();
  uint32_t ref_loc = start_loc;
  auto curr_type = var_type;
  if (ref->opcode() == spv::Op::OpAccessChain ||
      ref->opcode() == spv::Op::OpInBoundsAccessChain) {
    live_mgr->AnalyzeAccessChainLoc(ref, &curr_type, &ref_loc, &no_loc,
                                    is_patch, /* input */ false);
  }
  if (no_loc || AnyLocsAreLive(ref_loc, live_mgr->GetLocSize(curr_type)))
    return;
  // Kill all stores based on this reference
  KillAllStoresOfRef(ref);
}

void EliminateDeadOutputStoresPass::KillAllDeadStoresOfBuiltinRef(
    Instruction* ref, Instruction* var) {
  auto deco_mgr = context()->get_decoration_mgr();
  auto def_use_mgr = context()->get_def_use_mgr();
  auto type_mgr = context()->get_type_mgr();
  auto live_mgr = context()->get_liveness_mgr();
  // Search for builtin decoration of base variable
  uint32_t builtin = uint32_t(spv::BuiltIn::Max);
  auto var_id = var->result_id();
  (void)deco_mgr->WhileEachDecoration(
      var_id, uint32_t(spv::Decoration::BuiltIn),
      [&builtin](const Instruction& deco) {
        assert(deco.opcode() == spv::Op::OpDecorate && "unexpected decoration");
        builtin = deco.GetSingleWordInOperand(kOpDecorateBuiltInLiteralInIdx);
        return false;
      });
  // If analyzed builtin and not live, kill stores.
  if (builtin != uint32_t(spv::BuiltIn::Max)) {
    if (live_mgr->IsAnalyzedBuiltin(builtin) && !IsLiveBuiltin(builtin))
      KillAllStoresOfRef(ref);
    return;
  }
  // Search for builtin decoration on indexed member
  auto ref_op = ref->opcode();
  if (ref_op != spv::Op::OpAccessChain &&
      ref_op != spv::Op::OpInBoundsAccessChain) {
    return;
  }
  uint32_t in_idx = kOpAccessChainIdx0InIdx;
  analysis::Type* var_type = type_mgr->GetType(var->type_id());
  analysis::Pointer* ptr_type = var_type->AsPointer();
  auto curr_type = ptr_type->pointee_type();
  auto arr_type = curr_type->AsArray();
  if (arr_type) {
    curr_type = arr_type->element_type();
    ++in_idx;
  }
  auto str_type = curr_type->AsStruct();
  auto str_type_id = type_mgr->GetId(str_type);
  auto member_idx_id = ref->GetSingleWordInOperand(in_idx);
  auto member_idx_inst = def_use_mgr->GetDef(member_idx_id);
  assert(member_idx_inst->opcode() == spv::Op::OpConstant &&
         "unexpected non-constant index");
  auto ac_idx = member_idx_inst->GetSingleWordInOperand(kOpConstantValueInIdx);
  (void)deco_mgr->WhileEachDecoration(
      str_type_id, uint32_t(spv::Decoration::BuiltIn),
      [ac_idx, &builtin](const Instruction& deco) {
        assert(deco.opcode() == spv::Op::OpMemberDecorate &&
               "unexpected decoration");
        auto deco_idx =
            deco.GetSingleWordInOperand(kOpDecorateMemberMemberInIdx);
        if (deco_idx == ac_idx) {
          builtin =
              deco.GetSingleWordInOperand(kOpDecorateMemberBuiltInLiteralInIdx);
          return false;
        }
        return true;
      });
  assert(builtin != uint32_t(spv::BuiltIn::Max) && "builtin not found");
  // If analyzed builtin and not live, kill stores.
  if (live_mgr->IsAnalyzedBuiltin(builtin) && !IsLiveBuiltin(builtin))
    KillAllStoresOfRef(ref);
}

Pass::Status EliminateDeadOutputStoresPass::DoDeadOutputStoreElimination() {
  // Current implementation only supports vert, tesc, tese, geom shaders
  auto stage = context()->GetStage();
  if (stage != spv::ExecutionModel::Vertex &&
      stage != spv::ExecutionModel::TessellationControl &&
      stage != spv::ExecutionModel::TessellationEvaluation &&
      stage != spv::ExecutionModel::Geometry)
    return Status::Failure;
  InitializeElimination();
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::DecorationManager* deco_mgr = context()->get_decoration_mgr();
  // Process all output variables
  for (auto& var : context()->types_values()) {
    if (var.opcode() != spv::Op::OpVariable) {
      continue;
    }
    analysis::Type* var_type = type_mgr->GetType(var.type_id());
    analysis::Pointer* ptr_type = var_type->AsPointer();
    if (ptr_type->storage_class() != spv::StorageClass::Output) {
      continue;
    }
    // If builtin decoration on variable, process as builtin.
    auto var_id = var.result_id();
    bool is_builtin = false;
    if (deco_mgr->HasDecoration(var_id, uint32_t(spv::Decoration::BuiltIn))) {
      is_builtin = true;
    } else {
      // If interface block with builtin members, process as builtin.
      // Strip off outer array type if present.
      auto curr_type = ptr_type->pointee_type();
      auto arr_type = curr_type->AsArray();
      if (arr_type) curr_type = arr_type->element_type();
      auto str_type = curr_type->AsStruct();
      if (str_type) {
        auto str_type_id = type_mgr->GetId(str_type);
        if (deco_mgr->HasDecoration(str_type_id,
                                    uint32_t(spv::Decoration::BuiltIn)))
          is_builtin = true;
      }
    }
    // For each store or access chain using var, if dead builtin or all its
    // locations are dead, kill store or all access chain's stores
    def_use_mgr->ForEachUser(
        var_id, [this, &var, is_builtin](Instruction* user) {
          auto op = user->opcode();
          if (op == spv::Op::OpEntryPoint || op == spv::Op::OpName ||
              op == spv::Op::OpDecorate)
            return;
          if (is_builtin)
            KillAllDeadStoresOfBuiltinRef(user, &var);
          else
            KillAllDeadStoresOfLocRef(user, &var);
        });
  }
  for (auto& kinst : kill_list_) context()->KillInst(kinst);

  return kill_list_.empty() ? Status::SuccessWithoutChange
                            : Status::SuccessWithChange;
}

}  // namespace opt
}  // namespace spvtools
