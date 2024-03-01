// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
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

#ifndef SOURCE_OPT_AGGRESSIVE_DEAD_CODE_ELIM_PASS_H_
#define SOURCE_OPT_AGGRESSIVE_DEAD_CODE_ELIM_PASS_H_

#include <algorithm>
#include <list>
#include <map>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "source/opt/basic_block.h"
#include "source/opt/def_use_manager.h"
#include "source/opt/mem_pass.h"
#include "source/opt/module.h"
#include "source/util/bit_vector.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class AggressiveDCEPass : public MemPass {
  using cbb_ptr = const BasicBlock*;

 public:
  using GetBlocksFunction =
      std::function<std::vector<BasicBlock*>*(const BasicBlock*)>;

  AggressiveDCEPass(bool preserve_interface = false,
                    bool remove_outputs = false)
      : preserve_interface_(preserve_interface),
        remove_outputs_(remove_outputs) {}

  const char* name() const override { return "eliminate-dead-code-aggressive"; }
  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDefUse |
           IRContext::kAnalysisInstrToBlockMapping |
           IRContext::kAnalysisConstants | IRContext::kAnalysisTypes;
  }

 private:
  // Preserve entry point interface if true. All variables in interface
  // will be marked live and will not be eliminated. This mode is needed by
  // GPU-Assisted Validation instrumentation where a change in the interface
  // is not allowed.
  bool preserve_interface_;

  // Output variables can be removed from the interface if this is true.
  // This is safe if the caller knows that the corresponding input variable
  // in the following shader has been removed. It is false by default.
  bool remove_outputs_;

  // Return true if |varId| is a variable of |storageClass|. |varId| must either
  // be 0 or the result of an instruction.
  bool IsVarOfStorage(uint32_t varId, spv::StorageClass storageClass);

  // Return true if the instance of the variable |varId| can only be access in
  // |func|.  For example, a function scope variable, or a private variable
  // where |func| is an entry point with no function calls.
  bool IsLocalVar(uint32_t varId, Function* func);

  // Return true if |inst| is marked live.
  bool IsLive(const Instruction* inst) const {
    return live_insts_.Get(inst->unique_id());
  }

  // Adds entry points, execution modes and workgroup size decorations to the
  // worklist for processing with the first function.
  void InitializeModuleScopeLiveInstructions();

  // Add |inst| to worklist_ and live_insts_.
  void AddToWorklist(Instruction* inst) {
    if (!live_insts_.Set(inst->unique_id())) {
      worklist_.push(inst);
    }
  }

  // Add all store instruction which use |ptrId|, directly or indirectly,
  // to the live instruction worklist.
  void AddStores(Function* func, uint32_t ptrId);

  // Initialize extensions allowlist
  void InitExtensions();

  // Return true if all extensions in this module are supported by this pass.
  bool AllExtensionsSupported() const;

  // Returns true if the target of |inst| is dead.  An instruction is dead if
  // its result id is used in decoration or debug instructions only. |inst| is
  // assumed to be OpName, OpMemberName or an annotation instruction.
  bool IsTargetDead(Instruction* inst);

  // If |varId| is local, mark all stores of varId as live.
  void ProcessLoad(Function* func, uint32_t varId);

  // Add branch to |labelId| to end of block |bp|.
  void AddBranch(uint32_t labelId, BasicBlock* bp);

  // Add all break and continue branches in the construct associated with
  // |mergeInst| to worklist if not already live
  void AddBreaksAndContinuesToWorklist(Instruction* mergeInst);

  // Eliminates dead debug2 and annotation instructions. Marks dead globals for
  // removal (e.g. types, constants and variables).
  bool ProcessGlobalValues();

  // Erases functions that are unreachable from the entry points of the module.
  bool EliminateDeadFunctions();

  // For function |func|, mark all Stores to non-function-scope variables
  // and block terminating instructions as live. Recursively mark the values
  // they use. When complete, mark any non-live instructions to be deleted.
  // Returns true if the function has been modified.
  //
  // Note: This function does not delete useless control structures. All
  // existing control structures will remain. This can leave not-insignificant
  // sequences of ultimately useless code.
  // TODO(): Remove useless control constructs.
  bool AggressiveDCE(Function* func);

  Pass::Status ProcessImpl();

  // Adds instructions which must be kept because of they have side-effects
  // that ADCE cannot model to the work list.
  void InitializeWorkList(Function* func,
                          std::list<BasicBlock*>& structured_order);

  // Process each instruction in the work list by marking any instruction that
  // that it depends on as live, and adding it to the work list.  The work list
  // will be empty at the end.
  void ProcessWorkList(Function* func);

  // Kills any instructions in |func| that have not been marked as live.
  bool KillDeadInstructions(const Function* func,
                            std::list<BasicBlock*>& structured_order);

  // Adds the instructions that define the operands of |inst| to the work list.
  void AddOperandsToWorkList(const Instruction* inst);

  // Marks all of the labels and branch that inst requires as live.
  void MarkBlockAsLive(Instruction* inst);

  // Marks any variables from which |inst| may require data as live.
  void MarkLoadedVariablesAsLive(Function* func, Instruction* inst);

  // Returns the id of the variable that |ptr_id| point to.  |ptr_id| must be a
  // value whose type is a pointer.
  uint32_t GetVariableId(uint32_t ptr_id);

  // Returns all of the ids for the variables from which |inst| will load data.
  std::vector<uint32_t> GetLoadedVariables(Instruction* inst);

  // Returns all of the ids for the variables from which |inst| will load data.
  // The opcode of |inst| must be  OpFunctionCall.
  std::vector<uint32_t> GetLoadedVariablesFromFunctionCall(
      const Instruction* inst);

  // Returns the id of the variable from which |inst| will load data. |inst|
  // must not be an OpFunctionCall.  Returns 0 if no data is read or the
  // variable cannot be determined.  Note that in logical addressing mode the
  // latter is not possible for function and private storage class because there
  // cannot be variable pointers pointing to those storage classes.
  uint32_t GetLoadedVariableFromNonFunctionCalls(Instruction* inst);

  // Adds all decorations of |inst| to the work list.
  void AddDecorationsToWorkList(const Instruction* inst);

  // Adds DebugScope instruction associated with |inst| to the work list.
  void AddDebugScopeToWorkList(const Instruction* inst);

  // Adds all debug instruction associated with |inst| to the work list.
  void AddDebugInstructionsToWorkList(const Instruction* inst);

  // Marks all of the OpFunctionParameter instructions in |func| as live.
  void MarkFunctionParameterAsLive(const Function* func);

  // Returns the terminator instruction in the header for the innermost
  // construct that contains |blk|.  Returns nullptr if no such header exists.
  Instruction* GetHeaderBranch(BasicBlock* blk);

  // Returns the header for the innermost construct that contains |blk|.  A loop
  // header will be its own header.  Returns nullptr if no such header exists.
  BasicBlock* GetHeaderBlock(BasicBlock* blk) const;

  // Returns the same as |GetHeaderBlock| except if |blk| is a loop header it
  // will return the header of the next enclosing construct.  Returns nullptr if
  // no such header exists.
  Instruction* GetBranchForNextHeader(BasicBlock* blk);

  // Returns the merge instruction in the same basic block as |inst|.  Returns
  // nullptr if one does not exist.
  Instruction* GetMergeInstruction(Instruction* inst);

  // Returns true if |bb| is in the construct with header |header_block|.
  bool BlockIsInConstruct(BasicBlock* header_block, BasicBlock* bb);

  // Returns true if |func| is an entry point that does not have any function
  // calls.
  bool IsEntryPointWithNoCalls(Function* func);

  // Returns true if |func| is an entry point.
  bool IsEntryPoint(Function* func);

  // Returns true if |func| contains a function call.
  bool HasCall(Function* func);

  // Marks the first block, which is the entry block, in |func| as live.
  void MarkFirstBlockAsLive(Function* func);

  // Adds an OpUnreachable instruction at the end of |block|.
  void AddUnreachable(BasicBlock*& block);

  // Marks the OpLoopMerge and the terminator in |basic_block| as live if
  // |basic_block| is a loop header.
  void MarkLoopConstructAsLiveIfLoopHeader(BasicBlock* basic_block);

  // The cached results for |IsEntryPointWithNoCalls|.  It maps the function's
  // result id to the return value.
  std::unordered_map<uint32_t, bool> entry_point_with_no_calls_cache_;

  // Live Instruction Worklist.  An instruction is added to this list
  // if it might have a side effect, either directly or indirectly.
  // If we don't know, then add it to this list.  Instructions are
  // removed from this list as the algorithm traces side effects,
  // building up the live instructions set |live_insts_|.
  std::queue<Instruction*> worklist_;

  // Live Instructions
  utils::BitVector live_insts_;

  // Live Local Variables
  std::unordered_set<uint32_t> live_local_vars_;

  // List of instructions to delete. Deletion is delayed until debug and
  // annotation instructions are processed.
  std::vector<Instruction*> to_kill_;

  // Extensions supported by this pass.
  std::unordered_set<std::string> extensions_allowlist_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_AGGRESSIVE_DEAD_CODE_ELIM_PASS_H_
