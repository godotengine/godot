// Copyright (c) 2020-2022 Google LLC
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

#include "source/opt/debug_info_manager.h"

#include <cassert>

#include "source/opt/ir_context.h"

// Constants for OpenCL.DebugInfo.100 & NonSemantic.Shader.DebugInfo.100
// extension instructions.

namespace spvtools {
namespace opt {
namespace analysis {
namespace {
constexpr uint32_t kOpLineOperandLineIndex = 1;
constexpr uint32_t kLineOperandIndexDebugFunction = 7;
constexpr uint32_t kLineOperandIndexDebugLexicalBlock = 5;
constexpr uint32_t kLineOperandIndexDebugLine = 5;
constexpr uint32_t kDebugFunctionOperandFunctionIndex = 13;
constexpr uint32_t kDebugFunctionDefinitionOperandDebugFunctionIndex = 4;
constexpr uint32_t kDebugFunctionDefinitionOperandOpFunctionIndex = 5;
constexpr uint32_t kDebugFunctionOperandParentIndex = 9;
constexpr uint32_t kDebugTypeCompositeOperandParentIndex = 9;
constexpr uint32_t kDebugLexicalBlockOperandParentIndex = 7;
constexpr uint32_t kDebugInlinedAtOperandInlinedIndex = 6;
constexpr uint32_t kDebugExpressOperandOperationIndex = 4;
constexpr uint32_t kDebugDeclareOperandLocalVariableIndex = 4;
constexpr uint32_t kDebugDeclareOperandVariableIndex = 5;
constexpr uint32_t kDebugValueOperandExpressionIndex = 6;
constexpr uint32_t kDebugOperationOperandOperationIndex = 4;
constexpr uint32_t kOpVariableOperandStorageClassIndex = 2;
constexpr uint32_t kDebugLocalVariableOperandParentIndex = 9;
constexpr uint32_t kExtInstInstructionInIdx = 1;
constexpr uint32_t kDebugGlobalVariableOperandFlagsIndex = 12;
constexpr uint32_t kDebugLocalVariableOperandFlagsIndex = 10;

void SetInlinedOperand(Instruction* dbg_inlined_at, uint32_t inlined_operand) {
  assert(dbg_inlined_at);
  assert(dbg_inlined_at->GetCommonDebugOpcode() ==
         CommonDebugInfoDebugInlinedAt);
  if (dbg_inlined_at->NumOperands() <= kDebugInlinedAtOperandInlinedIndex) {
    dbg_inlined_at->AddOperand(
        {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {inlined_operand}});
  } else {
    dbg_inlined_at->SetOperand(kDebugInlinedAtOperandInlinedIndex,
                               {inlined_operand});
  }
}

uint32_t GetInlinedOperand(Instruction* dbg_inlined_at) {
  assert(dbg_inlined_at);
  assert(dbg_inlined_at->GetCommonDebugOpcode() ==
         CommonDebugInfoDebugInlinedAt);
  if (dbg_inlined_at->NumOperands() <= kDebugInlinedAtOperandInlinedIndex)
    return kNoInlinedAt;
  return dbg_inlined_at->GetSingleWordOperand(
      kDebugInlinedAtOperandInlinedIndex);
}

bool IsEmptyDebugExpression(Instruction* instr) {
  return (instr->GetCommonDebugOpcode() == CommonDebugInfoDebugExpression) &&
         instr->NumOperands() == kDebugExpressOperandOperationIndex;
}

}  // namespace

DebugInfoManager::DebugInfoManager(IRContext* c) : context_(c) {
  AnalyzeDebugInsts(*c->module());
}

uint32_t DebugInfoManager::GetDbgSetImportId() {
  uint32_t setId =
      context()->get_feature_mgr()->GetExtInstImportId_OpenCL100DebugInfo();
  if (setId == 0) {
    setId =
        context()->get_feature_mgr()->GetExtInstImportId_Shader100DebugInfo();
  }
  return setId;
}

Instruction* DebugInfoManager::GetDbgInst(uint32_t id) {
  auto dbg_inst_it = id_to_dbg_inst_.find(id);
  return dbg_inst_it == id_to_dbg_inst_.end() ? nullptr : dbg_inst_it->second;
}

void DebugInfoManager::RegisterDbgInst(Instruction* inst) {
  assert(inst->NumInOperands() != 0 &&
         (GetDbgSetImportId() == inst->GetInOperand(0).words[0]) &&
         "Given instruction is not a debug instruction");
  id_to_dbg_inst_[inst->result_id()] = inst;
}

void DebugInfoManager::RegisterDbgFunction(Instruction* inst) {
  if (inst->GetOpenCL100DebugOpcode() == OpenCLDebugInfo100DebugFunction) {
    auto fn_id = inst->GetSingleWordOperand(kDebugFunctionOperandFunctionIndex);
    // Do not register function that has been optimized away.
    auto fn_inst = GetDbgInst(fn_id);
    if (fn_inst != nullptr) {
      assert(GetDbgInst(fn_id)->GetOpenCL100DebugOpcode() ==
             OpenCLDebugInfo100DebugInfoNone);
      return;
    }
    assert(
        fn_id_to_dbg_fn_.find(fn_id) == fn_id_to_dbg_fn_.end() &&
        "Register DebugFunction for a function that already has DebugFunction");
    fn_id_to_dbg_fn_[fn_id] = inst;
  } else if (inst->GetShader100DebugOpcode() ==
             NonSemanticShaderDebugInfo100DebugFunctionDefinition) {
    auto fn_id = inst->GetSingleWordOperand(
        kDebugFunctionDefinitionOperandOpFunctionIndex);
    auto fn_inst = GetDbgInst(inst->GetSingleWordOperand(
        kDebugFunctionDefinitionOperandDebugFunctionIndex));
    assert(fn_inst && fn_inst->GetShader100DebugOpcode() ==
                          NonSemanticShaderDebugInfo100DebugFunction);
    assert(fn_id_to_dbg_fn_.find(fn_id) == fn_id_to_dbg_fn_.end() &&
           "Register DebugFunctionDefinition for a function that already has "
           "DebugFunctionDefinition");
    fn_id_to_dbg_fn_[fn_id] = fn_inst;
  } else {
    assert(false && "inst is not a DebugFunction");
  }
}

void DebugInfoManager::RegisterDbgDeclare(uint32_t var_id,
                                          Instruction* dbg_declare) {
  assert(dbg_declare->GetCommonDebugOpcode() == CommonDebugInfoDebugDeclare ||
         dbg_declare->GetCommonDebugOpcode() == CommonDebugInfoDebugValue);
  auto dbg_decl_itr = var_id_to_dbg_decl_.find(var_id);
  if (dbg_decl_itr == var_id_to_dbg_decl_.end()) {
    var_id_to_dbg_decl_[var_id] = {dbg_declare};
  } else {
    dbg_decl_itr->second.insert(dbg_declare);
  }
}

// Create new constant directly into global value area, bypassing the
// Constant manager. This is used when the DefUse or Constant managers
// are invalid and cannot be regenerated due to the module being in an
// inconsistent state e.g. in the middle of significant modification
// such as inlining. Invalidate Constant and DefUse managers if used.
uint32_t AddNewConstInGlobals(IRContext* context, uint32_t const_value) {
  uint32_t id = context->TakeNextId();
  std::unique_ptr<Instruction> new_const(new Instruction(
      context, spv::Op::OpConstant, context->get_type_mgr()->GetUIntTypeId(),
      id,
      {
          {spv_operand_type_t::SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER,
           {const_value}},
      }));
  context->module()->AddGlobalValue(std::move(new_const));
  context->InvalidateAnalyses(IRContext::kAnalysisConstants);
  context->InvalidateAnalyses(IRContext::kAnalysisDefUse);
  return id;
}

uint32_t DebugInfoManager::CreateDebugInlinedAt(const Instruction* line,
                                                const DebugScope& scope) {
  uint32_t setId = GetDbgSetImportId();

  if (setId == 0) return kNoInlinedAt;

  spv_operand_type_t line_number_type =
      spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER;

  // In NonSemantic.Shader.DebugInfo.100, all constants are IDs of OpConstant,
  // not literals.
  if (setId ==
      context()->get_feature_mgr()->GetExtInstImportId_Shader100DebugInfo())
    line_number_type = spv_operand_type_t::SPV_OPERAND_TYPE_ID;

  uint32_t line_number = 0;
  if (line == nullptr) {
    auto* lexical_scope_inst = GetDbgInst(scope.GetLexicalScope());
    if (lexical_scope_inst == nullptr) return kNoInlinedAt;
    CommonDebugInfoInstructions debug_opcode =
        lexical_scope_inst->GetCommonDebugOpcode();
    switch (debug_opcode) {
      case CommonDebugInfoDebugFunction:
        line_number = lexical_scope_inst->GetSingleWordOperand(
            kLineOperandIndexDebugFunction);
        break;
      case CommonDebugInfoDebugLexicalBlock:
        line_number = lexical_scope_inst->GetSingleWordOperand(
            kLineOperandIndexDebugLexicalBlock);
        break;
      case CommonDebugInfoDebugTypeComposite:
      case CommonDebugInfoDebugCompilationUnit:
        assert(false &&
               "DebugTypeComposite and DebugCompilationUnit are lexical "
               "scopes, but we inline functions into a function or a block "
               "of a function, not into a struct/class or a global scope.");
        break;
      default:
        assert(false &&
               "Unreachable. a debug extension instruction for a "
               "lexical scope must be DebugFunction, DebugTypeComposite, "
               "DebugLexicalBlock, or DebugCompilationUnit.");
        break;
    }
  } else {
    if (line->opcode() == spv::Op::OpLine) {
      line_number = line->GetSingleWordOperand(kOpLineOperandLineIndex);
    } else if (line->GetShader100DebugOpcode() ==
               NonSemanticShaderDebugInfo100DebugLine) {
      line_number = line->GetSingleWordOperand(kLineOperandIndexDebugLine);
    } else {
      assert(false &&
             "Unreachable. A line instruction must be OpLine or DebugLine");
    }

    // If we need the line number as an ID, generate that constant now.
    // If Constant or DefUse managers are invalid, generate constant
    // directly into the global value section of the module; do not
    // use Constant manager which may attempt to invoke building of the
    // DefUse manager which cannot be done during inlining. The extra
    // constants that may be generated here is likely not significant
    // and will likely be cleaned up in later passes.
    if (line_number_type == spv_operand_type_t::SPV_OPERAND_TYPE_ID &&
        line->opcode() == spv::Op::OpLine) {
      if (!context()->AreAnalysesValid(IRContext::Analysis::kAnalysisDefUse) ||
          !context()->AreAnalysesValid(IRContext::Analysis::kAnalysisConstants))
        line_number = AddNewConstInGlobals(context(), line_number);
      else
        line_number =
            context()->get_constant_mgr()->GetUIntConstId(line_number);
    }
  }

  uint32_t result_id = context()->TakeNextId();
  std::unique_ptr<Instruction> inlined_at(new Instruction(
      context(), spv::Op::OpExtInst, context()->get_type_mgr()->GetVoidTypeId(),
      result_id,
      {
          {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {setId}},
          {spv_operand_type_t::SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER,
           {static_cast<uint32_t>(CommonDebugInfoDebugInlinedAt)}},
          {line_number_type, {line_number}},
          {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {scope.GetLexicalScope()}},
      }));
  // |scope| already has DebugInlinedAt. We put the existing DebugInlinedAt
  // into the Inlined operand of this new DebugInlinedAt.
  if (scope.GetInlinedAt() != kNoInlinedAt) {
    inlined_at->AddOperand(
        {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {scope.GetInlinedAt()}});
  }
  RegisterDbgInst(inlined_at.get());
  if (context()->AreAnalysesValid(IRContext::Analysis::kAnalysisDefUse))
    context()->get_def_use_mgr()->AnalyzeInstDefUse(inlined_at.get());
  context()->module()->AddExtInstDebugInfo(std::move(inlined_at));
  return result_id;
}

DebugScope DebugInfoManager::BuildDebugScope(
    const DebugScope& callee_instr_scope,
    DebugInlinedAtContext* inlined_at_ctx) {
  return DebugScope(callee_instr_scope.GetLexicalScope(),
                    BuildDebugInlinedAtChain(callee_instr_scope.GetInlinedAt(),
                                             inlined_at_ctx));
}

uint32_t DebugInfoManager::BuildDebugInlinedAtChain(
    uint32_t callee_inlined_at, DebugInlinedAtContext* inlined_at_ctx) {
  if (inlined_at_ctx->GetScopeOfCallInstruction().GetLexicalScope() ==
      kNoDebugScope)
    return kNoInlinedAt;

  // Reuse the already generated DebugInlinedAt chain if exists.
  uint32_t already_generated_chain_head_id =
      inlined_at_ctx->GetDebugInlinedAtChain(callee_inlined_at);
  if (already_generated_chain_head_id != kNoInlinedAt) {
    return already_generated_chain_head_id;
  }

  const uint32_t new_dbg_inlined_at_id =
      CreateDebugInlinedAt(inlined_at_ctx->GetLineOfCallInstruction(),
                           inlined_at_ctx->GetScopeOfCallInstruction());
  if (new_dbg_inlined_at_id == kNoInlinedAt) return kNoInlinedAt;

  if (callee_inlined_at == kNoInlinedAt) {
    inlined_at_ctx->SetDebugInlinedAtChain(kNoInlinedAt, new_dbg_inlined_at_id);
    return new_dbg_inlined_at_id;
  }

  uint32_t chain_head_id = kNoInlinedAt;
  uint32_t chain_iter_id = callee_inlined_at;
  Instruction* last_inlined_at_in_chain = nullptr;
  do {
    Instruction* new_inlined_at_in_chain = CloneDebugInlinedAt(
        chain_iter_id, /* insert_before */ last_inlined_at_in_chain);
    assert(new_inlined_at_in_chain != nullptr);

    // Set DebugInlinedAt of the new scope as the head of the chain.
    if (chain_head_id == kNoInlinedAt)
      chain_head_id = new_inlined_at_in_chain->result_id();

    // Previous DebugInlinedAt of the chain must point to the new
    // DebugInlinedAt as its Inlined operand to build a recursive
    // chain.
    if (last_inlined_at_in_chain != nullptr) {
      SetInlinedOperand(last_inlined_at_in_chain,
                        new_inlined_at_in_chain->result_id());
    }
    last_inlined_at_in_chain = new_inlined_at_in_chain;

    chain_iter_id = GetInlinedOperand(new_inlined_at_in_chain);
  } while (chain_iter_id != kNoInlinedAt);

  // Put |new_dbg_inlined_at_id| into the end of the chain.
  SetInlinedOperand(last_inlined_at_in_chain, new_dbg_inlined_at_id);

  // Keep the new chain information that will be reused it.
  inlined_at_ctx->SetDebugInlinedAtChain(callee_inlined_at, chain_head_id);
  return chain_head_id;
}

Instruction* DebugInfoManager::GetDebugOperationWithDeref() {
  if (deref_operation_ != nullptr) return deref_operation_;

  uint32_t result_id = context()->TakeNextId();
  std::unique_ptr<Instruction> deref_operation;

  if (context()->get_feature_mgr()->GetExtInstImportId_OpenCL100DebugInfo()) {
    deref_operation = std::unique_ptr<Instruction>(new Instruction(
        context(), spv::Op::OpExtInst,
        context()->get_type_mgr()->GetVoidTypeId(), result_id,
        {
            {SPV_OPERAND_TYPE_ID, {GetDbgSetImportId()}},
            {SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER,
             {static_cast<uint32_t>(OpenCLDebugInfo100DebugOperation)}},
            {SPV_OPERAND_TYPE_CLDEBUG100_DEBUG_OPERATION,
             {static_cast<uint32_t>(OpenCLDebugInfo100Deref)}},
        }));
  } else {
    uint32_t deref_id = context()->get_constant_mgr()->GetUIntConstId(
        NonSemanticShaderDebugInfo100Deref);

    deref_operation = std::unique_ptr<Instruction>(
        new Instruction(context(), spv::Op::OpExtInst,
                        context()->get_type_mgr()->GetVoidTypeId(), result_id,
                        {
                            {SPV_OPERAND_TYPE_ID, {GetDbgSetImportId()}},
                            {SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER,
                             {static_cast<uint32_t>(
                                 NonSemanticShaderDebugInfo100DebugOperation)}},
                            {SPV_OPERAND_TYPE_ID, {deref_id}},
                        }));
  }

  // Add to the front of |ext_inst_debuginfo_|.
  deref_operation_ =
      context()->module()->ext_inst_debuginfo_begin()->InsertBefore(
          std::move(deref_operation));

  RegisterDbgInst(deref_operation_);
  if (context()->AreAnalysesValid(IRContext::Analysis::kAnalysisDefUse))
    context()->get_def_use_mgr()->AnalyzeInstDefUse(deref_operation_);
  return deref_operation_;
}

Instruction* DebugInfoManager::DerefDebugExpression(Instruction* dbg_expr) {
  assert(dbg_expr->GetCommonDebugOpcode() == CommonDebugInfoDebugExpression);
  std::unique_ptr<Instruction> deref_expr(dbg_expr->Clone(context()));
  deref_expr->SetResultId(context()->TakeNextId());
  deref_expr->InsertOperand(
      kDebugExpressOperandOperationIndex,
      {SPV_OPERAND_TYPE_ID, {GetDebugOperationWithDeref()->result_id()}});
  auto* deref_expr_instr =
      context()->ext_inst_debuginfo_end()->InsertBefore(std::move(deref_expr));
  AnalyzeDebugInst(deref_expr_instr);
  if (context()->AreAnalysesValid(IRContext::Analysis::kAnalysisDefUse))
    context()->get_def_use_mgr()->AnalyzeInstDefUse(deref_expr_instr);
  return deref_expr_instr;
}

Instruction* DebugInfoManager::GetDebugInfoNone() {
  if (debug_info_none_inst_ != nullptr) return debug_info_none_inst_;

  uint32_t result_id = context()->TakeNextId();
  std::unique_ptr<Instruction> dbg_info_none_inst(new Instruction(
      context(), spv::Op::OpExtInst, context()->get_type_mgr()->GetVoidTypeId(),
      result_id,
      {
          {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {GetDbgSetImportId()}},
          {spv_operand_type_t::SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER,
           {static_cast<uint32_t>(CommonDebugInfoDebugInfoNone)}},
      }));

  // Add to the front of |ext_inst_debuginfo_|.
  debug_info_none_inst_ =
      context()->module()->ext_inst_debuginfo_begin()->InsertBefore(
          std::move(dbg_info_none_inst));

  RegisterDbgInst(debug_info_none_inst_);
  if (context()->AreAnalysesValid(IRContext::Analysis::kAnalysisDefUse))
    context()->get_def_use_mgr()->AnalyzeInstDefUse(debug_info_none_inst_);
  return debug_info_none_inst_;
}

Instruction* DebugInfoManager::GetEmptyDebugExpression() {
  if (empty_debug_expr_inst_ != nullptr) return empty_debug_expr_inst_;

  uint32_t result_id = context()->TakeNextId();
  std::unique_ptr<Instruction> empty_debug_expr(new Instruction(
      context(), spv::Op::OpExtInst, context()->get_type_mgr()->GetVoidTypeId(),
      result_id,
      {
          {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {GetDbgSetImportId()}},
          {spv_operand_type_t::SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER,
           {static_cast<uint32_t>(CommonDebugInfoDebugExpression)}},
      }));

  // Add to the front of |ext_inst_debuginfo_|.
  empty_debug_expr_inst_ =
      context()->module()->ext_inst_debuginfo_begin()->InsertBefore(
          std::move(empty_debug_expr));

  RegisterDbgInst(empty_debug_expr_inst_);
  if (context()->AreAnalysesValid(IRContext::Analysis::kAnalysisDefUse))
    context()->get_def_use_mgr()->AnalyzeInstDefUse(empty_debug_expr_inst_);
  return empty_debug_expr_inst_;
}

Instruction* DebugInfoManager::GetDebugInlinedAt(uint32_t dbg_inlined_at_id) {
  auto* inlined_at = GetDbgInst(dbg_inlined_at_id);
  if (inlined_at == nullptr) return nullptr;
  if (inlined_at->GetCommonDebugOpcode() != CommonDebugInfoDebugInlinedAt) {
    return nullptr;
  }
  return inlined_at;
}

Instruction* DebugInfoManager::CloneDebugInlinedAt(uint32_t clone_inlined_at_id,
                                                   Instruction* insert_before) {
  auto* inlined_at = GetDebugInlinedAt(clone_inlined_at_id);
  if (inlined_at == nullptr) return nullptr;
  std::unique_ptr<Instruction> new_inlined_at(inlined_at->Clone(context()));
  new_inlined_at->SetResultId(context()->TakeNextId());
  RegisterDbgInst(new_inlined_at.get());
  if (context()->AreAnalysesValid(IRContext::Analysis::kAnalysisDefUse))
    context()->get_def_use_mgr()->AnalyzeInstDefUse(new_inlined_at.get());
  if (insert_before != nullptr)
    return insert_before->InsertBefore(std::move(new_inlined_at));
  return context()->module()->ext_inst_debuginfo_end()->InsertBefore(
      std::move(new_inlined_at));
}

bool DebugInfoManager::IsVariableDebugDeclared(uint32_t variable_id) {
  auto dbg_decl_itr = var_id_to_dbg_decl_.find(variable_id);
  return dbg_decl_itr != var_id_to_dbg_decl_.end();
}

bool DebugInfoManager::KillDebugDeclares(uint32_t variable_id) {
  bool modified = false;
  auto dbg_decl_itr = var_id_to_dbg_decl_.find(variable_id);
  if (dbg_decl_itr != var_id_to_dbg_decl_.end()) {
    // We intentionally copy the list of DebugDeclare instructions because
    // context()->KillInst(dbg_decl) will update |var_id_to_dbg_decl_|. If we
    // directly use |dbg_decl_itr->second|, it accesses a dangling pointer.
    auto copy_dbg_decls = dbg_decl_itr->second;

    for (auto* dbg_decl : copy_dbg_decls) {
      context()->KillInst(dbg_decl);
      modified = true;
    }
    var_id_to_dbg_decl_.erase(dbg_decl_itr);
  }
  return modified;
}

uint32_t DebugInfoManager::GetParentScope(uint32_t child_scope) {
  auto dbg_scope_itr = id_to_dbg_inst_.find(child_scope);
  assert(dbg_scope_itr != id_to_dbg_inst_.end());
  CommonDebugInfoInstructions debug_opcode =
      dbg_scope_itr->second->GetCommonDebugOpcode();
  uint32_t parent_scope = kNoDebugScope;
  switch (debug_opcode) {
    case CommonDebugInfoDebugFunction:
      parent_scope = dbg_scope_itr->second->GetSingleWordOperand(
          kDebugFunctionOperandParentIndex);
      break;
    case CommonDebugInfoDebugLexicalBlock:
      parent_scope = dbg_scope_itr->second->GetSingleWordOperand(
          kDebugLexicalBlockOperandParentIndex);
      break;
    case CommonDebugInfoDebugTypeComposite:
      parent_scope = dbg_scope_itr->second->GetSingleWordOperand(
          kDebugTypeCompositeOperandParentIndex);
      break;
    case CommonDebugInfoDebugCompilationUnit:
      // DebugCompilationUnit does not have a parent scope.
      break;
    default:
      assert(false &&
             "Unreachable. A debug scope instruction must be "
             "DebugFunction, DebugTypeComposite, DebugLexicalBlock, "
             "or DebugCompilationUnit.");
      break;
  }
  return parent_scope;
}

bool DebugInfoManager::IsAncestorOfScope(uint32_t scope, uint32_t ancestor) {
  uint32_t ancestor_scope_itr = scope;
  while (ancestor_scope_itr != kNoDebugScope) {
    if (ancestor == ancestor_scope_itr) return true;
    ancestor_scope_itr = GetParentScope(ancestor_scope_itr);
  }
  return false;
}

bool DebugInfoManager::IsDeclareVisibleToInstr(Instruction* dbg_declare,
                                               Instruction* scope) {
  assert(dbg_declare != nullptr);
  assert(scope != nullptr);

  std::vector<uint32_t> scope_ids;
  if (scope->opcode() == spv::Op::OpPhi) {
    scope_ids.push_back(scope->GetDebugScope().GetLexicalScope());
    for (uint32_t i = 0; i < scope->NumInOperands(); i += 2) {
      auto* value = context()->get_def_use_mgr()->GetDef(
          scope->GetSingleWordInOperand(i));
      if (value != nullptr)
        scope_ids.push_back(value->GetDebugScope().GetLexicalScope());
    }
  } else {
    scope_ids.push_back(scope->GetDebugScope().GetLexicalScope());
  }

  uint32_t dbg_local_var_id =
      dbg_declare->GetSingleWordOperand(kDebugDeclareOperandLocalVariableIndex);
  auto dbg_local_var_itr = id_to_dbg_inst_.find(dbg_local_var_id);
  assert(dbg_local_var_itr != id_to_dbg_inst_.end());
  uint32_t decl_scope_id = dbg_local_var_itr->second->GetSingleWordOperand(
      kDebugLocalVariableOperandParentIndex);

  // If the scope of DebugDeclare is an ancestor scope of the instruction's
  // scope, the local variable is visible to the instruction.
  for (uint32_t scope_id : scope_ids) {
    if (scope_id != kNoDebugScope &&
        IsAncestorOfScope(scope_id, decl_scope_id)) {
      return true;
    }
  }
  return false;
}

bool DebugInfoManager::AddDebugValueForVariable(Instruction* scope_and_line,
                                                uint32_t variable_id,
                                                uint32_t value_id,
                                                Instruction* insert_pos) {
  assert(scope_and_line != nullptr);

  auto dbg_decl_itr = var_id_to_dbg_decl_.find(variable_id);
  if (dbg_decl_itr == var_id_to_dbg_decl_.end()) return false;

  bool modified = false;
  for (auto* dbg_decl_or_val : dbg_decl_itr->second) {
    // Avoid inserting the new DebugValue between OpPhi or OpVariable
    // instructions.
    Instruction* insert_before = insert_pos->NextNode();
    while (insert_before->opcode() == spv::Op::OpPhi ||
           insert_before->opcode() == spv::Op::OpVariable) {
      insert_before = insert_before->NextNode();
    }
    modified |= AddDebugValueForDecl(dbg_decl_or_val, value_id, insert_before,
                                     scope_and_line) != nullptr;
  }
  return modified;
}

Instruction* DebugInfoManager::AddDebugValueForDecl(
    Instruction* dbg_decl, uint32_t value_id, Instruction* insert_before,
    Instruction* scope_and_line) {
  if (dbg_decl == nullptr || !IsDebugDeclare(dbg_decl)) return nullptr;

  std::unique_ptr<Instruction> dbg_val(dbg_decl->Clone(context()));
  dbg_val->SetResultId(context()->TakeNextId());
  dbg_val->SetInOperand(kExtInstInstructionInIdx, {CommonDebugInfoDebugValue});
  dbg_val->SetOperand(kDebugDeclareOperandVariableIndex, {value_id});
  dbg_val->SetOperand(kDebugValueOperandExpressionIndex,
                      {GetEmptyDebugExpression()->result_id()});
  dbg_val->UpdateDebugInfoFrom(scope_and_line);

  auto* added_dbg_val = insert_before->InsertBefore(std::move(dbg_val));
  AnalyzeDebugInst(added_dbg_val);
  if (context()->AreAnalysesValid(IRContext::Analysis::kAnalysisDefUse))
    context()->get_def_use_mgr()->AnalyzeInstDefUse(added_dbg_val);
  if (context()->AreAnalysesValid(
          IRContext::Analysis::kAnalysisInstrToBlockMapping)) {
    auto insert_blk = context()->get_instr_block(insert_before);
    context()->set_instr_block(added_dbg_val, insert_blk);
  }
  return added_dbg_val;
}

uint32_t DebugInfoManager::GetVulkanDebugOperation(Instruction* inst) {
  assert(inst->GetShader100DebugOpcode() ==
             NonSemanticShaderDebugInfo100DebugOperation &&
         "inst must be Vulkan DebugOperation");
  return context()
      ->get_constant_mgr()
      ->GetConstantFromInst(context()->get_def_use_mgr()->GetDef(
          inst->GetSingleWordOperand(kDebugOperationOperandOperationIndex)))
      ->GetU32();
}

uint32_t DebugInfoManager::GetVariableIdOfDebugValueUsedForDeclare(
    Instruction* inst) {
  if (inst->GetCommonDebugOpcode() != CommonDebugInfoDebugValue) return 0;

  auto* expr =
      GetDbgInst(inst->GetSingleWordOperand(kDebugValueOperandExpressionIndex));
  if (expr == nullptr) return 0;
  if (expr->NumOperands() != kDebugExpressOperandOperationIndex + 1) return 0;

  auto* operation = GetDbgInst(
      expr->GetSingleWordOperand(kDebugExpressOperandOperationIndex));
  if (operation == nullptr) return 0;

  // OpenCL.DebugInfo.100 contains a literal for the operation, Vulkan uses an
  // OpConstant.
  if (inst->IsOpenCL100DebugInstr()) {
    if (operation->GetSingleWordOperand(kDebugOperationOperandOperationIndex) !=
        OpenCLDebugInfo100Deref) {
      return 0;
    }
  } else {
    uint32_t operation_const = GetVulkanDebugOperation(operation);
    if (operation_const != NonSemanticShaderDebugInfo100Deref) {
      return 0;
    }
  }

  uint32_t var_id =
      inst->GetSingleWordOperand(kDebugDeclareOperandVariableIndex);
  if (!context()->AreAnalysesValid(IRContext::Analysis::kAnalysisDefUse)) {
    assert(false &&
           "Checking a DebugValue can be used for declare needs DefUseManager");
    return 0;
  }

  auto* var = context()->get_def_use_mgr()->GetDef(var_id);
  if (var->opcode() == spv::Op::OpVariable &&
      spv::StorageClass(
          var->GetSingleWordOperand(kOpVariableOperandStorageClassIndex)) ==
          spv::StorageClass::Function) {
    return var_id;
  }
  return 0;
}

bool DebugInfoManager::IsDebugDeclare(Instruction* instr) {
  if (!instr->IsCommonDebugInstr()) return false;
  return instr->GetCommonDebugOpcode() == CommonDebugInfoDebugDeclare ||
         GetVariableIdOfDebugValueUsedForDeclare(instr) != 0;
}

void DebugInfoManager::ReplaceAllUsesInDebugScopeWithPredicate(
    uint32_t before, uint32_t after,
    const std::function<bool(Instruction*)>& predicate) {
  auto scope_id_to_users_itr = scope_id_to_users_.find(before);
  if (scope_id_to_users_itr != scope_id_to_users_.end()) {
    for (Instruction* inst : scope_id_to_users_itr->second) {
      if (predicate(inst)) inst->UpdateLexicalScope(after);
    }
    scope_id_to_users_[after] = scope_id_to_users_itr->second;
    scope_id_to_users_.erase(scope_id_to_users_itr);
  }
  auto inlinedat_id_to_users_itr = inlinedat_id_to_users_.find(before);
  if (inlinedat_id_to_users_itr != inlinedat_id_to_users_.end()) {
    for (Instruction* inst : inlinedat_id_to_users_itr->second) {
      if (predicate(inst)) inst->UpdateDebugInlinedAt(after);
    }
    inlinedat_id_to_users_[after] = inlinedat_id_to_users_itr->second;
    inlinedat_id_to_users_.erase(inlinedat_id_to_users_itr);
  }
}

void DebugInfoManager::ClearDebugScopeAndInlinedAtUses(Instruction* inst) {
  auto scope_id_to_users_itr = scope_id_to_users_.find(inst->result_id());
  if (scope_id_to_users_itr != scope_id_to_users_.end()) {
    scope_id_to_users_.erase(scope_id_to_users_itr);
  }
  auto inlinedat_id_to_users_itr =
      inlinedat_id_to_users_.find(inst->result_id());
  if (inlinedat_id_to_users_itr != inlinedat_id_to_users_.end()) {
    inlinedat_id_to_users_.erase(inlinedat_id_to_users_itr);
  }
}

void DebugInfoManager::AnalyzeDebugInst(Instruction* inst) {
  if (inst->GetDebugScope().GetLexicalScope() != kNoDebugScope) {
    auto& users = scope_id_to_users_[inst->GetDebugScope().GetLexicalScope()];
    users.insert(inst);
  }
  if (inst->GetDebugInlinedAt() != kNoInlinedAt) {
    auto& users = inlinedat_id_to_users_[inst->GetDebugInlinedAt()];
    users.insert(inst);
  }

  if (!inst->IsCommonDebugInstr()) return;

  RegisterDbgInst(inst);

  if (inst->GetOpenCL100DebugOpcode() == OpenCLDebugInfo100DebugFunction ||
      inst->GetShader100DebugOpcode() ==
          NonSemanticShaderDebugInfo100DebugFunctionDefinition) {
    RegisterDbgFunction(inst);
  }

  if (deref_operation_ == nullptr &&
      inst->GetOpenCL100DebugOpcode() == OpenCLDebugInfo100DebugOperation &&
      inst->GetSingleWordOperand(kDebugOperationOperandOperationIndex) ==
          OpenCLDebugInfo100Deref) {
    deref_operation_ = inst;
  }

  if (deref_operation_ == nullptr &&
      inst->GetShader100DebugOpcode() ==
          NonSemanticShaderDebugInfo100DebugOperation) {
    uint32_t operation_const = GetVulkanDebugOperation(inst);
    if (operation_const == NonSemanticShaderDebugInfo100Deref) {
      deref_operation_ = inst;
    }
  }

  if (debug_info_none_inst_ == nullptr &&
      inst->GetCommonDebugOpcode() == CommonDebugInfoDebugInfoNone) {
    debug_info_none_inst_ = inst;
  }

  if (empty_debug_expr_inst_ == nullptr && IsEmptyDebugExpression(inst)) {
    empty_debug_expr_inst_ = inst;
  }

  if (inst->GetCommonDebugOpcode() == CommonDebugInfoDebugDeclare) {
    uint32_t var_id =
        inst->GetSingleWordOperand(kDebugDeclareOperandVariableIndex);
    RegisterDbgDeclare(var_id, inst);
  }

  if (uint32_t var_id = GetVariableIdOfDebugValueUsedForDeclare(inst)) {
    RegisterDbgDeclare(var_id, inst);
  }
}

void DebugInfoManager::ConvertDebugGlobalToLocalVariable(
    Instruction* dbg_global_var, Instruction* local_var) {
  if (dbg_global_var->GetCommonDebugOpcode() !=
      CommonDebugInfoDebugGlobalVariable) {
    return;
  }
  assert(local_var->opcode() == spv::Op::OpVariable ||
         local_var->opcode() == spv::Op::OpFunctionParameter);

  // Convert |dbg_global_var| to DebugLocalVariable
  dbg_global_var->SetInOperand(kExtInstInstructionInIdx,
                               {CommonDebugInfoDebugLocalVariable});
  auto flags = dbg_global_var->GetSingleWordOperand(
      kDebugGlobalVariableOperandFlagsIndex);
  for (uint32_t i = dbg_global_var->NumInOperands() - 1;
       i >= kDebugLocalVariableOperandFlagsIndex; --i) {
    dbg_global_var->RemoveOperand(i);
  }
  dbg_global_var->SetOperand(kDebugLocalVariableOperandFlagsIndex, {flags});
  context()->ForgetUses(dbg_global_var);
  context()->AnalyzeUses(dbg_global_var);

  // Create a DebugDeclare
  std::unique_ptr<Instruction> new_dbg_decl(new Instruction(
      context(), spv::Op::OpExtInst, context()->get_type_mgr()->GetVoidTypeId(),
      context()->TakeNextId(),
      {
          {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {GetDbgSetImportId()}},
          {spv_operand_type_t::SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER,
           {static_cast<uint32_t>(CommonDebugInfoDebugDeclare)}},
          {spv_operand_type_t::SPV_OPERAND_TYPE_ID,
           {dbg_global_var->result_id()}},
          {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {local_var->result_id()}},
          {spv_operand_type_t::SPV_OPERAND_TYPE_ID,
           {GetEmptyDebugExpression()->result_id()}},
      }));
  // Must insert after all OpVariables in block
  Instruction* insert_before = local_var;
  while (insert_before->opcode() == spv::Op::OpVariable)
    insert_before = insert_before->NextNode();
  auto* added_dbg_decl = insert_before->InsertBefore(std::move(new_dbg_decl));
  if (context()->AreAnalysesValid(IRContext::Analysis::kAnalysisDefUse))
    context()->get_def_use_mgr()->AnalyzeInstDefUse(added_dbg_decl);
  if (context()->AreAnalysesValid(
          IRContext::Analysis::kAnalysisInstrToBlockMapping)) {
    auto insert_blk = context()->get_instr_block(local_var);
    context()->set_instr_block(added_dbg_decl, insert_blk);
  }
}

void DebugInfoManager::AnalyzeDebugInsts(Module& module) {
  deref_operation_ = nullptr;
  debug_info_none_inst_ = nullptr;
  empty_debug_expr_inst_ = nullptr;
  module.ForEachInst([this](Instruction* cpi) { AnalyzeDebugInst(cpi); });

  // Move |empty_debug_expr_inst_| to the beginning of the debug instruction
  // list.
  if (empty_debug_expr_inst_ != nullptr &&
      empty_debug_expr_inst_->PreviousNode() != nullptr &&
      empty_debug_expr_inst_->PreviousNode()->IsCommonDebugInstr()) {
    empty_debug_expr_inst_->InsertBefore(
        &*context()->module()->ext_inst_debuginfo_begin());
  }

  // Move |debug_info_none_inst_| to the beginning of the debug instruction
  // list.
  if (debug_info_none_inst_ != nullptr &&
      debug_info_none_inst_->PreviousNode() != nullptr &&
      debug_info_none_inst_->PreviousNode()->IsCommonDebugInstr()) {
    debug_info_none_inst_->InsertBefore(
        &*context()->module()->ext_inst_debuginfo_begin());
  }
}

void DebugInfoManager::ClearDebugInfo(Instruction* instr) {
  auto scope_id_to_users_itr =
      scope_id_to_users_.find(instr->GetDebugScope().GetLexicalScope());
  if (scope_id_to_users_itr != scope_id_to_users_.end()) {
    scope_id_to_users_itr->second.erase(instr);
  }
  auto inlinedat_id_to_users_itr =
      inlinedat_id_to_users_.find(instr->GetDebugInlinedAt());
  if (inlinedat_id_to_users_itr != inlinedat_id_to_users_.end()) {
    inlinedat_id_to_users_itr->second.erase(instr);
  }

  if (instr == nullptr || !instr->IsCommonDebugInstr()) {
    return;
  }

  id_to_dbg_inst_.erase(instr->result_id());

  if (instr->GetOpenCL100DebugOpcode() == OpenCLDebugInfo100DebugFunction) {
    auto fn_id =
        instr->GetSingleWordOperand(kDebugFunctionOperandFunctionIndex);
    fn_id_to_dbg_fn_.erase(fn_id);
  }
  if (instr->GetShader100DebugOpcode() ==
      NonSemanticShaderDebugInfo100DebugFunctionDefinition) {
    auto fn_id = instr->GetSingleWordOperand(
        kDebugFunctionDefinitionOperandOpFunctionIndex);
    fn_id_to_dbg_fn_.erase(fn_id);
  }

  if (instr->GetCommonDebugOpcode() == CommonDebugInfoDebugDeclare ||
      instr->GetCommonDebugOpcode() == CommonDebugInfoDebugValue) {
    auto var_or_value_id =
        instr->GetSingleWordOperand(kDebugDeclareOperandVariableIndex);
    auto dbg_decl_itr = var_id_to_dbg_decl_.find(var_or_value_id);
    if (dbg_decl_itr != var_id_to_dbg_decl_.end()) {
      dbg_decl_itr->second.erase(instr);
    }
  }

  if (deref_operation_ == instr) {
    deref_operation_ = nullptr;
    for (auto dbg_instr_itr = context()->module()->ext_inst_debuginfo_begin();
         dbg_instr_itr != context()->module()->ext_inst_debuginfo_end();
         ++dbg_instr_itr) {
      // OpenCL.DebugInfo.100 contains the operation as a literal operand, in
      // Vulkan it's referenced as an OpConstant.
      if (instr != &*dbg_instr_itr &&
          dbg_instr_itr->GetOpenCL100DebugOpcode() ==
              OpenCLDebugInfo100DebugOperation &&
          dbg_instr_itr->GetSingleWordOperand(
              kDebugOperationOperandOperationIndex) ==
              OpenCLDebugInfo100Deref) {
        deref_operation_ = &*dbg_instr_itr;
        break;
      } else if (instr != &*dbg_instr_itr &&
                 dbg_instr_itr->GetShader100DebugOpcode() ==
                     NonSemanticShaderDebugInfo100DebugOperation) {
        uint32_t operation_const = GetVulkanDebugOperation(&*dbg_instr_itr);
        if (operation_const == NonSemanticShaderDebugInfo100Deref) {
          deref_operation_ = &*dbg_instr_itr;
          break;
        }
      }
    }
  }

  if (debug_info_none_inst_ == instr) {
    debug_info_none_inst_ = nullptr;
    for (auto dbg_instr_itr = context()->module()->ext_inst_debuginfo_begin();
         dbg_instr_itr != context()->module()->ext_inst_debuginfo_end();
         ++dbg_instr_itr) {
      if (instr != &*dbg_instr_itr && dbg_instr_itr->GetCommonDebugOpcode() ==
                                          CommonDebugInfoDebugInfoNone) {
        debug_info_none_inst_ = &*dbg_instr_itr;
        break;
      }
    }
  }

  if (empty_debug_expr_inst_ == instr) {
    empty_debug_expr_inst_ = nullptr;
    for (auto dbg_instr_itr = context()->module()->ext_inst_debuginfo_begin();
         dbg_instr_itr != context()->module()->ext_inst_debuginfo_end();
         ++dbg_instr_itr) {
      if (instr != &*dbg_instr_itr && IsEmptyDebugExpression(&*dbg_instr_itr)) {
        empty_debug_expr_inst_ = &*dbg_instr_itr;
        break;
      }
    }
  }
}

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools
