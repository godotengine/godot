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

#ifndef SOURCE_OPT_DEBUG_INFO_MANAGER_H_
#define SOURCE_OPT_DEBUG_INFO_MANAGER_H_

#include <set>
#include <unordered_map>
#include <unordered_set>

#include "source/opt/instruction.h"
#include "source/opt/module.h"

namespace spvtools {
namespace opt {
namespace analysis {

// When an instruction of a callee function is inlined to its caller function,
// we need the line and the scope information of the function call instruction
// to generate DebugInlinedAt. This class keeps the data. For multiple inlining
// of a single instruction, we have to create multiple DebugInlinedAt
// instructions as a chain. This class keeps the information of the generated
// DebugInlinedAt chains to reduce the number of chains.
class DebugInlinedAtContext {
 public:
  explicit DebugInlinedAtContext(Instruction* call_inst)
      : call_inst_line_(call_inst->dbg_line_inst()),
        call_inst_scope_(call_inst->GetDebugScope()) {}

  const Instruction* GetLineOfCallInstruction() { return call_inst_line_; }
  const DebugScope& GetScopeOfCallInstruction() { return call_inst_scope_; }
  // Puts the DebugInlinedAt chain that is generated for the callee instruction
  // whose DebugInlinedAt of DebugScope is |callee_instr_inlined_at| into
  // |callee_inlined_at2chain_|.
  void SetDebugInlinedAtChain(uint32_t callee_instr_inlined_at,
                              uint32_t chain_head_id) {
    callee_inlined_at2chain_[callee_instr_inlined_at] = chain_head_id;
  }
  // Gets the DebugInlinedAt chain from |callee_inlined_at2chain_|.
  uint32_t GetDebugInlinedAtChain(uint32_t callee_instr_inlined_at) {
    auto chain_itr = callee_inlined_at2chain_.find(callee_instr_inlined_at);
    if (chain_itr != callee_inlined_at2chain_.end()) return chain_itr->second;
    return kNoInlinedAt;
  }

 private:
  // The line information of the function call instruction that will be
  // replaced by the callee function.
  const Instruction* call_inst_line_;

  // The scope information of the function call instruction that will be
  // replaced by the callee function.
  const DebugScope call_inst_scope_;

  // Map from DebugInlinedAt ids of callee to head ids of new generated
  // DebugInlinedAt chain.
  std::unordered_map<uint32_t, uint32_t> callee_inlined_at2chain_;
};

// A class for analyzing, managing, and creating OpenCL.DebugInfo.100 and
// NonSemantic.Shader.DebugInfo.100 extension instructions.
class DebugInfoManager {
 public:
  // Constructs a debug information manager from the given |context|.
  DebugInfoManager(IRContext* context);

  DebugInfoManager(const DebugInfoManager&) = delete;
  DebugInfoManager(DebugInfoManager&&) = delete;
  DebugInfoManager& operator=(const DebugInfoManager&) = delete;
  DebugInfoManager& operator=(DebugInfoManager&&) = delete;

  friend bool operator==(const DebugInfoManager&, const DebugInfoManager&);
  friend bool operator!=(const DebugInfoManager& lhs,
                         const DebugInfoManager& rhs) {
    return !(lhs == rhs);
  }

  // Analyzes DebugInfo instruction |dbg_inst|.
  void AnalyzeDebugInst(Instruction* dbg_inst);

  // Creates new DebugInlinedAt and returns its id. Its line operand is the
  // line number of |line| if |line| is not nullptr. Otherwise, its line operand
  // is the line number of lexical scope of |scope|. Its Scope and Inlined
  // operands are Scope and Inlined of |scope|.
  uint32_t CreateDebugInlinedAt(const Instruction* line,
                                const DebugScope& scope);

  // Clones DebugExpress instruction |dbg_expr| and add Deref Operation
  // in the front of the Operation list of |dbg_expr|.
  Instruction* DerefDebugExpression(Instruction* dbg_expr);

  // Returns a DebugInfoNone instruction.
  Instruction* GetDebugInfoNone();

  // Returns DebugInlinedAt whose id is |dbg_inlined_at_id|. If it does not
  // exist or it is not a DebugInlinedAt instruction, return nullptr.
  Instruction* GetDebugInlinedAt(uint32_t dbg_inlined_at_id);

  // Returns DebugFunction whose Function operand is |fn_id|. If it does not
  // exist, return nullptr.
  Instruction* GetDebugFunction(uint32_t fn_id) {
    auto dbg_fn_it = fn_id_to_dbg_fn_.find(fn_id);
    return dbg_fn_it == fn_id_to_dbg_fn_.end() ? nullptr : dbg_fn_it->second;
  }

  // Clones DebugInlinedAt whose id is |clone_inlined_at_id|. If
  // |clone_inlined_at_id| is not an id of DebugInlinedAt, returns nullptr.
  // If |insert_before| is given, inserts the new DebugInlinedAt before it.
  // Otherwise, inserts the new DebugInlinedAt into the debug instruction
  // section of the module.
  Instruction* CloneDebugInlinedAt(uint32_t clone_inlined_at_id,
                                   Instruction* insert_before = nullptr);

  // Returns the debug scope corresponding to an inlining instruction in the
  // scope |callee_instr_scope| into |inlined_at_ctx|. Generates all new
  // debug instructions needed to represent the scope.
  DebugScope BuildDebugScope(const DebugScope& callee_instr_scope,
                             DebugInlinedAtContext* inlined_at_ctx);

  // Returns DebugInlinedAt corresponding to inlining an instruction, which
  // was inlined at |callee_inlined_at|, into |inlined_at_ctx|. Generates all
  // new debug instructions needed to represent the DebugInlinedAt.
  uint32_t BuildDebugInlinedAtChain(uint32_t callee_inlined_at,
                                    DebugInlinedAtContext* inlined_at_ctx);

  // Returns true if there is a debug declaration instruction whose
  // 'Local Variable' operand is |variable_id|.
  bool IsVariableDebugDeclared(uint32_t variable_id);

  // Kills all debug declaration instructions with Deref whose 'Local Variable'
  // operand is |variable_id|. Returns whether it kills an instruction or not.
  bool KillDebugDeclares(uint32_t variable_id);

  // Generates a DebugValue instruction with value |value_id| for every local
  // variable that is in the scope of |scope_and_line| and whose memory is
  // |variable_id| and inserts it after the instruction |insert_pos|.
  // Returns whether a DebugValue is added or not.
  bool AddDebugValueForVariable(Instruction* scope_and_line,
                                uint32_t variable_id, uint32_t value_id,
                                Instruction* insert_pos);

  // Creates a DebugValue for DebugDeclare |dbg_decl| and inserts it before
  // |insert_before|. The new DebugValue has the same line and scope as
  // |scope_and_line|, or no scope and line information if |scope_and_line|
  // is nullptr. The new DebugValue has the same operands as DebugDeclare
  // but it uses |value_id| for the value. Returns the created DebugValue,
  // or nullptr if fails to create one.
  Instruction* AddDebugValueForDecl(Instruction* dbg_decl, uint32_t value_id,
                                    Instruction* insert_before,
                                    Instruction* scope_and_line);

  // Erases |instr| from data structures of this class.
  void ClearDebugInfo(Instruction* instr);

  // Return the opcode for the Vulkan DebugOperation inst
  uint32_t GetVulkanDebugOperation(Instruction* inst);

  // Returns the id of Value operand if |inst| is DebugValue who has Deref
  // operation and its Value operand is a result id of OpVariable with
  // Function storage class. Otherwise, returns 0.
  uint32_t GetVariableIdOfDebugValueUsedForDeclare(Instruction* inst);

  // Converts DebugGlobalVariable |dbg_global_var| to a DebugLocalVariable and
  // creates a DebugDeclare mapping the new DebugLocalVariable to |local_var|.
  void ConvertDebugGlobalToLocalVariable(Instruction* dbg_global_var,
                                         Instruction* local_var);

  // Returns true if |instr| is a debug declaration instruction.
  bool IsDebugDeclare(Instruction* instr);

  // Replace all uses of |before| id that is an operand of a DebugScope with
  // |after| id if those uses (instruction) return true for |predicate|.
  void ReplaceAllUsesInDebugScopeWithPredicate(
      uint32_t before, uint32_t after,
      const std::function<bool(Instruction*)>& predicate);

  // Removes uses of DebugScope |inst| from |scope_id_to_users_| or uses of
  // DebugInlinedAt |inst| from |inlinedat_id_to_users_|.
  void ClearDebugScopeAndInlinedAtUses(Instruction* inst);

 private:
  IRContext* context() { return context_; }

  // Analyzes DebugInfo instructions in the given |module| and
  // populates data structures in this class.
  void AnalyzeDebugInsts(Module& module);

  // Get the DebugInfo ExtInstImport Id, or 0 if no DebugInfo is available.
  uint32_t GetDbgSetImportId();

  // Returns the debug instruction whose id is |id|. Returns |nullptr| if one
  // does not exists.
  Instruction* GetDbgInst(uint32_t id);

  // Returns a DebugOperation instruction with OpCode Deref.
  Instruction* GetDebugOperationWithDeref();

  // Registers the debug instruction |inst| into |id_to_dbg_inst_| using id of
  // |inst| as a key.
  void RegisterDbgInst(Instruction* inst);

  // Register the DebugFunction instruction |inst|. The function referenced
  // in |inst| must not already be registered.
  void RegisterDbgFunction(Instruction* inst);

  // Register the DebugDeclare or DebugValue with Deref operation
  // |dbg_declare| into |var_id_to_dbg_decl_| using OpVariable id
  // |var_id| as a key.
  void RegisterDbgDeclare(uint32_t var_id, Instruction* dbg_declare);

  // Returns a DebugExpression instruction without Operation operands.
  Instruction* GetEmptyDebugExpression();

  // Returns true if a scope |ancestor| is |scope| or an ancestor scope
  // of |scope|.
  bool IsAncestorOfScope(uint32_t scope, uint32_t ancestor);

  // Returns true if the declaration of a local variable |dbg_declare|
  // is visible in the scope of an instruction |instr_scope_id|.
  bool IsDeclareVisibleToInstr(Instruction* dbg_declare, Instruction* scope);

  // Returns the parent scope of the scope |child_scope|.
  uint32_t GetParentScope(uint32_t child_scope);

  IRContext* context_;

  // Mapping from ids of DebugInfo extension instructions.
  // to their Instruction instances.
  std::unordered_map<uint32_t, Instruction*> id_to_dbg_inst_;

  // Mapping from function's ids to DebugFunction instructions whose
  // operand is the function.
  std::unordered_map<uint32_t, Instruction*> fn_id_to_dbg_fn_;

  // Orders Instruction* for use in associative containers (i.e. less than
  // ordering). Unique Id is used.
  typedef Instruction* InstPtr;
  struct InstPtrLess {
    bool operator()(const InstPtr& lhs, const InstPtr& rhs) const {
      return lhs->unique_id() < rhs->unique_id();
    }
  };

  // Mapping from variable or value ids to DebugDeclare or DebugValue
  // instructions whose operand is the variable or value.
  std::unordered_map<uint32_t, std::set<InstPtr, InstPtrLess>>
      var_id_to_dbg_decl_;

  // Mapping from DebugScope ids to users.
  std::unordered_map<uint32_t, std::unordered_set<Instruction*>>
      scope_id_to_users_;

  // Mapping from DebugInlinedAt ids to users.
  std::unordered_map<uint32_t, std::unordered_set<Instruction*>>
      inlinedat_id_to_users_;

  // DebugOperation whose OpCode is OpenCLDebugInfo100Deref.
  Instruction* deref_operation_;

  // DebugInfoNone instruction. We need only a single DebugInfoNone.
  // To reuse the existing one, we keep it using this member variable.
  Instruction* debug_info_none_inst_;

  // DebugExpression instruction without Operation operands. We need only
  // a single DebugExpression without Operation operands. To reuse the
  // existing one, we keep it using this member variable.
  Instruction* empty_debug_expr_inst_;
};

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_DEBUG_INFO_MANAGER_H_
