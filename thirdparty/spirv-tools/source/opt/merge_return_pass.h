// Copyright (c) 2017 Google Inc.
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

#ifndef SOURCE_OPT_MERGE_RETURN_PASS_H_
#define SOURCE_OPT_MERGE_RETURN_PASS_H_

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "source/opt/basic_block.h"
#include "source/opt/function.h"
#include "source/opt/mem_pass.h"

namespace spvtools {
namespace opt {

/*******************************************************************************
 *
 * Handling Structured Control Flow:
 *
 * Structured control flow guarantees that the CFG will converge at a given
 * point (the merge block). Within structured control flow, all blocks must be
 * post-dominated by the merge block, except return blocks and break blocks.
 * A break block is a block that branches to a containing construct's merge
 * block.
 *
 * Beyond this, we further assume that all unreachable blocks have been
 * cleaned up.  This means that the only unreachable blocks are those necessary
 * for valid structured control flow.
 *
 * Algorithm:
 *
 * If a return is encountered, it should record that: i) the function has
 * "returned" and ii) the value of the return. The return should be replaced
 * with a branch. If current block is not within structured control flow, this
 * is the final return. This block should branch to the new return block (its
 * direct successor). If the current block is within structured control flow,
 * the branch destination should be the innermost construct's merge.  This
 * merge will always exist because a single case switch is added around the
 * entire function. If the merge block produces any live values it will need to
 * be predicated. While the merge is nested in structured control flow, the
 * predication path should branch to the merge block of the inner-most loop
 * (or switch if no loop) it is contained in. Once structured control flow has
 * been exited, it will be at the merge of the single case switch, which will
 * simply return.
 *
 * In the final return block, the return value should be loaded and returned.
 * Memory promotion passes should be able to promote the newly introduced
 * variables ("has returned" and "return value").
 *
 * Predicating the Final Merge:
 *
 * At each merge block predication needs to be introduced (optimization: only if
 * that block produces value live beyond it). This needs to be done carefully.
 * The merge block should be split into multiple blocks.
 *
 *          1 (loop header)
 *        /   \
 * (ret) 2     3 (merge)
 *
 *         ||
 *         \/
 *
 *          0 (single case switch header)
 *          |
 *          1 (loop header)
 *         / \
 *        2  | (merge)
 *        \ /
 *         3' (merge)
 *        / \
 *        |  3 (original code in 3)
 *        \ /
 *   (ret) 4 (single case switch merge)
 *
 * In the above (simple) example, the return originally in |2| is passed through
 * the loop merge. That merge is predicated such that the old body of the block
 * is the else branch. The branch condition is based on the value of the "has
 * returned" variable.
 *
 ******************************************************************************/

// Documented in optimizer.hpp
class MergeReturnPass : public MemPass {
 public:
  MergeReturnPass()
      : function_(nullptr),
        return_flag_(nullptr),
        return_value_(nullptr),
        constant_true_(nullptr),
        final_return_block_(nullptr) {}

  const char* name() const override { return "merge-return"; }
  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisConstants | IRContext::kAnalysisTypes;
  }

 private:
  // This class is used to store the a break merge instruction and a current
  // merge instruction.  The intended use is to keep track of the block to
  // break to and the current innermost control flow construct merge block.
  class StructuredControlState {
   public:
    StructuredControlState(Instruction* break_merge, Instruction* merge)
        : break_merge_(break_merge), current_merge_(merge) {}

    bool InBreakable() const { return break_merge_; }
    bool InStructuredFlow() const { return CurrentMergeId() != 0; }

    uint32_t CurrentMergeId() const {
      return current_merge_ ? current_merge_->GetSingleWordInOperand(0u) : 0u;
    }

    uint32_t CurrentMergeHeader() const {
      return current_merge_ ? current_merge_->context()
                                  ->get_instr_block(current_merge_)
                                  ->id()
                            : 0;
    }

    uint32_t BreakMergeId() const {
      return break_merge_ ? break_merge_->GetSingleWordInOperand(0u) : 0u;
    }

    Instruction* BreakMergeInst() const { return break_merge_; }

   private:
    Instruction* break_merge_;
    Instruction* current_merge_;
  };

  // Returns all BasicBlocks terminated by OpReturn or OpReturnValue in
  // |function|.
  std::vector<BasicBlock*> CollectReturnBlocks(Function* function);

  // Creates a new basic block with a single return. If |function| returns a
  // value, a phi node is created to select the correct value to return.
  // Replaces old returns with an unconditional branch to the new block.
  void MergeReturnBlocks(Function* function,
                         const std::vector<BasicBlock*>& returnBlocks);

  // Generate and push new control flow state if |block| contains a merge.
  void GenerateState(BasicBlock* block);

  // Merges the return instruction in |function| so that it has a single return
  // statement.  It is assumed that |function| has structured control flow, and
  // that |return_blocks| is a list of all of the basic blocks in |function|
  // that have a return.
  bool ProcessStructured(Function* function,
                         const std::vector<BasicBlock*>& return_blocks);

  // Changes an OpReturn* or OpUnreachable instruction at the end of |block|
  // into a store to |return_flag_|, a store to |return_value_| (if necessary),
  // and a branch to the appropriate merge block.
  //
  // Is is assumed that |AddReturnValue| have already been called to created the
  // variable to store a return value if there is one.
  //
  // Note this will break the semantics.  To fix this, PredicateBlock will have
  // to be called on the merge block the branch targets.
  void ProcessStructuredBlock(BasicBlock* block);

  // Creates a variable used to store whether or not the control flow has
  // traversed a block that used to have a return.  A pointer to the instruction
  // declaring the variable is stored in |return_flag_|.
  void AddReturnFlag();

  // Creates the variable used to store the return value when passing through
  // a block that use to contain an OpReturnValue.
  void AddReturnValue();

  // Adds a store that stores true to |return_flag_| immediately before the
  // terminator of |block|. It is assumed that |AddReturnFlag| has already been
  // called.
  void RecordReturned(BasicBlock* block);

  // Adds an instruction that stores the value being returned in the
  // OpReturnValue in |block|.  The value is stored to |return_value_|, and the
  // store is placed before the OpReturnValue.
  //
  // If |block| does not contain an OpReturnValue, then this function has no
  // effect. If |block| contains an OpReturnValue, then |AddReturnValue| must
  // have already been called to create the variable to store to.
  void RecordReturnValue(BasicBlock* block);

  // Adds an unconditional branch in |block| that branches to |target|.  It also
  // adds stores to |return_flag_| and |return_value_| as needed.
  // |AddReturnFlag| and |AddReturnValue| must have already been called.
  void BranchToBlock(BasicBlock* block, uint32_t target);

  // For every basic block that is reachable from |return_block|, extra code is
  // added to jump around any code that should not be executed because the
  // original code would have already returned. This involves adding new
  // selections constructs to jump around these instructions.
  //
  // If new blocks that are created will be added to |order|.  This way a call
  // can traverse these new block in structured order.
  //
  // Returns true if successful.
  bool PredicateBlocks(BasicBlock* return_block,
                       std::unordered_set<BasicBlock*>* pSet,
                       std::list<BasicBlock*>* order);

  // Add a conditional branch at the start of |block| that either jumps to
  // the merge block of |break_merge_inst| or the original code in |block|
  // depending on the value in |return_flag_|.  The continue target in
  // |break_merge_inst| will be updated if needed.
  //
  // If new blocks that are created will be added to |order|.  This way a call
  // can traverse these new block in structured order.
  //
  // Returns true if successful.
  bool BreakFromConstruct(BasicBlock* block,
                          std::unordered_set<BasicBlock*>* predicated,
                          std::list<BasicBlock*>* order,
                          Instruction* break_merge_inst);

  // Add an |OpReturn| or |OpReturnValue| to the end of |block|.  If an
  // |OpReturnValue| is needed, the return value is loaded from |return_value_|.
  void CreateReturn(BasicBlock* block);

  // Creates a block at the end of the function that will become the single
  // return block at the end of the pass.
  void CreateReturnBlock();

  // Creates a Phi node in |merge_block| for the result of |inst|.
  // Any uses of the result of |inst| that are no longer
  // dominated by |inst|, are replaced with the result of the new |OpPhi|
  // instruction.
  void CreatePhiNodesForInst(BasicBlock* merge_block, Instruction& inst);

  // Add new phi nodes for any id that no longer dominate all of it uses.  A phi
  // node is added to a block |bb| for an id if the id is defined between the
  // original immediate dominator of |bb| and its new immediate dominator.  It
  // is assumed that at this point there are no unreachable blocks in the
  // control flow graph.
  void AddNewPhiNodes();

  // Creates any new phi nodes that are needed in |bb|.  |AddNewPhiNodes| must
  // have already been called on the original dominators of |bb|.
  void AddNewPhiNodes(BasicBlock* bb);

  // Records the terminator of immediate dominator for every basic block in
  // |function|.
  void RecordImmediateDominators(Function* function);

  // Modifies existing OpPhi instruction in |target| block to account for the
  // new edge from |new_source|.  The value for that edge will be an Undef.
  //
  // The CFG must not include the edge from |new_source| to |target| yet.
  void UpdatePhiNodes(BasicBlock* new_source, BasicBlock* target);

  StructuredControlState& CurrentState() { return state_.back(); }

  // Inserts |new_element| into |list| after the first occurrence of |element|.
  // |element| must be in |list| at least once.
  void InsertAfterElement(BasicBlock* element, BasicBlock* new_element,
                          std::list<BasicBlock*>* list);

  // Creates a single case switch around all of the executable code of the
  // current function where the switch and case value are both zero and the
  // default is the merge block. Returns after the switch is executed. Sets
  // |final_return_block_|.
  bool AddSingleCaseSwitchAroundFunction();

  // Creates a new basic block that branches to |header_label_id|.  Returns the
  // new basic block.  The block will be the second last basic block in the
  // function.
  BasicBlock* CreateContinueTarget(uint32_t header_label_id);

  // Creates a one case switch around the executable code of the function with
  // |merge_target| as the merge node.
  bool CreateSingleCaseSwitch(BasicBlock* merge_target);

  // Returns true if |function| has an unreachable block that is not a continue
  // target that simply branches back to the header, or a merge block containing
  // 1 instruction which is OpUnreachable.
  bool HasNontrivialUnreachableBlocks(Function* function);

  // A stack used to keep track of the break and current control flow construct
  // merge blocks.
  std::vector<StructuredControlState> state_;

  // The current function being transformed.
  Function* function_;

  // The |OpVariable| instruction defining a boolean variable used to keep track
  // of whether or not the function is trying to return.
  Instruction* return_flag_;

  // The |OpVariable| instruction defining a variabled to used to keep track of
  // the value that was returned when passing through a block that use to
  // contain an |OpReturnValue|.
  Instruction* return_value_;

  // The instruction defining the boolean constant true.
  Instruction* constant_true_;

  // The basic block that is suppose to become the contain the only return value
  // after processing the current function.
  BasicBlock* final_return_block_;

  // This is a map from a node to its original immediate dominator identified by
  // the terminator if that block.  We use the terminator because the block we
  // want may change if the block is split.
  std::unordered_map<BasicBlock*, Instruction*> original_dominator_;

  // A map from a basic block, bb, to the set of basic blocks which represent
  // the new edges that reach |bb|.
  std::unordered_map<BasicBlock*, std::set<uint32_t>> new_edges_;

  // Contains all return blocks that are merged. This is set is populated while
  // processing structured blocks and used to properly construct OpPhi
  // instructions.
  std::unordered_set<uint32_t> return_blocks_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_MERGE_RETURN_PASS_H_
