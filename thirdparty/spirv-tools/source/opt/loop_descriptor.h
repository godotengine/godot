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

#ifndef SOURCE_OPT_LOOP_DESCRIPTOR_H_
#define SOURCE_OPT_LOOP_DESCRIPTOR_H_

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "source/opt/basic_block.h"
#include "source/opt/dominator_analysis.h"
#include "source/opt/module.h"
#include "source/opt/tree_iterator.h"

namespace spvtools {
namespace opt {

class IRContext;
class CFG;
class LoopDescriptor;

// A class to represent and manipulate a loop in structured control flow.
class Loop {
  // The type used to represent nested child loops.
  using ChildrenList = std::vector<Loop*>;

 public:
  using iterator = ChildrenList::iterator;
  using const_iterator = ChildrenList::const_iterator;
  using BasicBlockListTy = std::unordered_set<uint32_t>;

  explicit Loop(IRContext* context)
      : context_(context),
        loop_header_(nullptr),
        loop_continue_(nullptr),
        loop_merge_(nullptr),
        loop_preheader_(nullptr),
        loop_latch_(nullptr),
        parent_(nullptr),
        loop_is_marked_for_removal_(false) {}

  Loop(IRContext* context, DominatorAnalysis* analysis, BasicBlock* header,
       BasicBlock* continue_target, BasicBlock* merge_target);

  // Iterators over the immediate sub-loops.
  inline iterator begin() { return nested_loops_.begin(); }
  inline iterator end() { return nested_loops_.end(); }
  inline const_iterator begin() const { return cbegin(); }
  inline const_iterator end() const { return cend(); }
  inline const_iterator cbegin() const { return nested_loops_.begin(); }
  inline const_iterator cend() const { return nested_loops_.end(); }

  // Returns the header (first basic block of the loop). This block contains the
  // OpLoopMerge instruction.
  inline BasicBlock* GetHeaderBlock() { return loop_header_; }
  inline const BasicBlock* GetHeaderBlock() const { return loop_header_; }
  inline void SetHeaderBlock(BasicBlock* header) { loop_header_ = header; }

  // Updates the OpLoopMerge instruction to reflect the current state of the
  // loop.
  inline void UpdateLoopMergeInst() {
    assert(GetHeaderBlock()->GetLoopMergeInst() &&
           "The loop is not structured");
    Instruction* merge_inst = GetHeaderBlock()->GetLoopMergeInst();
    merge_inst->SetInOperand(0, {GetMergeBlock()->id()});
  }

  // Returns the continue target basic block. This is the block designated as
  // the continue target by the OpLoopMerge instruction.
  inline BasicBlock* GetContinueBlock() { return loop_continue_; }
  inline const BasicBlock* GetContinueBlock() const { return loop_continue_; }

  // Returns the latch basic block (basic block that holds the back-edge).
  // These functions return nullptr if the loop is not structured (i.e. if it
  // has more than one backedge).
  inline BasicBlock* GetLatchBlock() { return loop_latch_; }
  inline const BasicBlock* GetLatchBlock() const { return loop_latch_; }

  // Sets |latch| as the loop unique block branching back to the header.
  // A latch block must have the following properties:
  //  - |latch| must be in the loop;
  //  - must be the only block branching back to the header block.
  void SetLatchBlock(BasicBlock* latch);

  // Sets |continue_block| as the continue block of the loop. This should be the
  // continue target of the OpLoopMerge and should dominate the latch block.
  void SetContinueBlock(BasicBlock* continue_block);

  // Returns the basic block which marks the end of the loop.
  // These functions return nullptr if the loop is not structured.
  inline BasicBlock* GetMergeBlock() { return loop_merge_; }
  inline const BasicBlock* GetMergeBlock() const { return loop_merge_; }
  // Sets |merge| as the loop merge block. A merge block must have the following
  // properties:
  //  - |merge| must not be in the loop;
  //  - all its predecessors must be in the loop.
  //  - it must not be already used as merge block.
  // If the loop has an OpLoopMerge in its header, this instruction is also
  // updated.
  void SetMergeBlock(BasicBlock* merge);

  // Returns the loop pre-header, nullptr means that the loop predecessor does
  // not qualify as a preheader.
  // The preheader is the unique predecessor that:
  //   - Dominates the loop header;
  //   - Has only the loop header as successor.
  inline BasicBlock* GetPreHeaderBlock() { return loop_preheader_; }

  // Returns the loop pre-header.
  inline const BasicBlock* GetPreHeaderBlock() const { return loop_preheader_; }
  // Sets |preheader| as the loop preheader block. A preheader block must have
  // the following properties:
  //  - |merge| must not be in the loop;
  //  - have an unconditional branch to the loop header.
  void SetPreHeaderBlock(BasicBlock* preheader);

  // Returns the loop pre-header, if there is no suitable preheader it will be
  // created.  Returns |nullptr| if it fails to create the preheader.
  BasicBlock* GetOrCreatePreHeaderBlock();

  // Returns true if this loop contains any nested loops.
  inline bool HasNestedLoops() const { return nested_loops_.size() != 0; }

  // Clears and fills |exit_blocks| with all basic blocks that are not in the
  // loop and has at least one predecessor in the loop.
  void GetExitBlocks(std::unordered_set<uint32_t>* exit_blocks) const;

  // Clears and fills |merging_blocks| with all basic blocks that are
  // post-dominated by the merge block. The merge block must exist.
  // The set |merging_blocks| will only contain the merge block if it is
  // unreachable.
  void GetMergingBlocks(std::unordered_set<uint32_t>* merging_blocks) const;

  // Returns true if the loop is in a Loop Closed SSA form.
  // In LCSSA form, all in-loop definitions are used in the loop or in phi
  // instructions in the loop exit blocks.
  bool IsLCSSA() const;

  // Returns the depth of this loop in the loop nest.
  // The outer-most loop has a depth of 1.
  inline size_t GetDepth() const {
    size_t lvl = 1;
    for (const Loop* loop = GetParent(); loop; loop = loop->GetParent()) lvl++;
    return lvl;
  }

  inline size_t NumImmediateChildren() const { return nested_loops_.size(); }

  inline bool HasChildren() const { return !nested_loops_.empty(); }
  // Adds |nested| as a nested loop of this loop. Automatically register |this|
  // as the parent of |nested|.
  inline void AddNestedLoop(Loop* nested) {
    assert(!nested->GetParent() && "The loop has another parent.");
    nested_loops_.push_back(nested);
    nested->SetParent(this);
  }

  inline Loop* GetParent() { return parent_; }
  inline const Loop* GetParent() const { return parent_; }

  inline bool HasParent() const { return parent_; }

  // Returns true if this loop is itself nested within another loop.
  inline bool IsNested() const { return parent_ != nullptr; }

  // Returns the set of all basic blocks contained within the loop. Will be all
  // BasicBlocks dominated by the header which are not also dominated by the
  // loop merge block.
  inline const BasicBlockListTy& GetBlocks() const {
    return loop_basic_blocks_;
  }

  // Returns true if the basic block |bb| is inside this loop.
  inline bool IsInsideLoop(const BasicBlock* bb) const {
    return IsInsideLoop(bb->id());
  }

  // Returns true if the basic block id |bb_id| is inside this loop.
  inline bool IsInsideLoop(uint32_t bb_id) const {
    return loop_basic_blocks_.count(bb_id);
  }

  // Returns true if the instruction |inst| is inside this loop.
  bool IsInsideLoop(Instruction* inst) const;

  // Adds the Basic Block |bb| to this loop and its parents.
  void AddBasicBlock(const BasicBlock* bb) { AddBasicBlock(bb->id()); }

  // Adds the Basic Block with |id| to this loop and its parents.
  void AddBasicBlock(uint32_t id) {
    for (Loop* loop = this; loop != nullptr; loop = loop->parent_) {
      loop->loop_basic_blocks_.insert(id);
    }
  }

  // Removes the Basic Block id |bb_id| from this loop and its parents.
  // It the user responsibility to make sure the removed block is not a merge,
  // header or continue block.
  void RemoveBasicBlock(uint32_t bb_id) {
    for (Loop* loop = this; loop != nullptr; loop = loop->parent_) {
      loop->loop_basic_blocks_.erase(bb_id);
    }
  }

  // Removes all the basic blocks from the set of basic blocks within the loop.
  // This does not affect any of the stored pointers to the header, preheader,
  // merge, or continue blocks.
  void ClearBlocks() { loop_basic_blocks_.clear(); }

  // Adds the Basic Block |bb| this loop and its parents.
  void AddBasicBlockToLoop(const BasicBlock* bb) {
    assert(IsBasicBlockInLoopSlow(bb) &&
           "Basic block does not belong to the loop");

    AddBasicBlock(bb);
  }

  // Returns the list of induction variables within the loop.
  void GetInductionVariables(std::vector<Instruction*>& inductions) const;

  // This function uses the |condition| to find the induction variable which is
  // used by the loop condition within the loop. This only works if the loop is
  // bound by a single condition and single induction variable.
  Instruction* FindConditionVariable(const BasicBlock* condition) const;

  // Returns the number of iterations within a loop when given the |induction|
  // variable and the loop |condition| check. It stores the found number of
  // iterations in the output parameter |iterations| and optionally, the step
  // value in |step_value| and the initial value of the induction variable in
  // |init_value|.
  bool FindNumberOfIterations(const Instruction* induction,
                              const Instruction* condition, size_t* iterations,
                              int64_t* step_amount = nullptr,
                              int64_t* init_value = nullptr) const;

  // Returns the value of the OpLoopMerge control operand as a bool. Loop
  // control can be None(0), Unroll(1), or DontUnroll(2). This function returns
  // true if it is set to Unroll.
  inline bool HasUnrollLoopControl() const {
    assert(loop_header_);
    if (!loop_header_->GetLoopMergeInst()) return false;

    return loop_header_->GetLoopMergeInst()->GetSingleWordOperand(2) == 1;
  }

  // Finds the conditional block with a branch to the merge and continue blocks
  // within the loop body.
  BasicBlock* FindConditionBlock() const;

  // Remove the child loop form this loop.
  inline void RemoveChildLoop(Loop* loop) {
    nested_loops_.erase(
        std::find(nested_loops_.begin(), nested_loops_.end(), loop));
    loop->SetParent(nullptr);
  }

  // Mark this loop to be removed later by a call to
  // LoopDescriptor::PostModificationCleanup.
  inline void MarkLoopForRemoval() { loop_is_marked_for_removal_ = true; }

  // Returns whether or not this loop has been marked for removal.
  inline bool IsMarkedForRemoval() const { return loop_is_marked_for_removal_; }

  // Returns true if all nested loops have been marked for removal.
  inline bool AreAllChildrenMarkedForRemoval() const {
    for (const Loop* child : nested_loops_) {
      if (!child->IsMarkedForRemoval()) {
        return false;
      }
    }
    return true;
  }

  // Checks if the loop contains any instruction that will prevent it from being
  // cloned. If the loop is structured, the merge construct is also considered.
  bool IsSafeToClone() const;

  // Sets the parent loop of this loop, that is, a loop which contains this loop
  // as a nested child loop.
  inline void SetParent(Loop* parent) { parent_ = parent; }

  // Returns true is the instruction is invariant and safe to move wrt loop
  bool ShouldHoistInstruction(IRContext* context, Instruction* inst);

  // Returns true if all operands of inst are in basic blocks not contained in
  // loop
  bool AreAllOperandsOutsideLoop(IRContext* context, Instruction* inst);

  // Extract the initial value from the |induction| variable and store it in
  // |value|. If the function couldn't find the initial value of |induction|
  // return false.
  bool GetInductionInitValue(const Instruction* induction,
                             int64_t* value) const;

  // Takes in a phi instruction |induction| and the loop |header| and returns
  // the step operation of the loop.
  Instruction* GetInductionStepOperation(const Instruction* induction) const;

  // Returns true if we can deduce the number of loop iterations in the step
  // operation |step|. IsSupportedCondition must also be true for the condition
  // instruction.
  bool IsSupportedStepOp(spv::Op step) const;

  // Returns true if we can deduce the number of loop iterations in the
  // condition operation |condition|. IsSupportedStepOp must also be true for
  // the step instruction.
  bool IsSupportedCondition(spv::Op condition) const;

  // Creates the list of the loop's basic block in structured order and store
  // the result in |ordered_loop_blocks|. If |include_pre_header| is true, the
  // pre-header block will also be included at the beginning of the list if it
  // exist. If |include_merge| is true, the merge block will also be included at
  // the end of the list if it exist.
  void ComputeLoopStructuredOrder(std::vector<BasicBlock*>* ordered_loop_blocks,
                                  bool include_pre_header = false,
                                  bool include_merge = false) const;

  // Given the loop |condition|, |initial_value|, |step_value|, the trip count
  // |number_of_iterations|, and the |unroll_factor| requested, get the new
  // condition value for the residual loop.
  static int64_t GetResidualConditionValue(spv::Op condition,
                                           int64_t initial_value,
                                           int64_t step_value,
                                           size_t number_of_iterations,
                                           size_t unroll_factor);

  // Returns the condition instruction for entry into the loop
  // Returns nullptr if it can't be found.
  Instruction* GetConditionInst() const;

  // Returns the context associated this loop.
  IRContext* GetContext() const { return context_; }

  // Looks at all the blocks with a branch to the header block to find one
  // which is also dominated by the loop continue block. This block is the latch
  // block. The specification mandates that this block should exist, therefore
  // this function will assert if it is not found.
  BasicBlock* FindLatchBlock();

 private:
  IRContext* context_;
  // The block which marks the start of the loop.
  BasicBlock* loop_header_;

  // The block which begins the body of the loop.
  BasicBlock* loop_continue_;

  // The block which marks the end of the loop.
  BasicBlock* loop_merge_;

  // The block immediately before the loop header.
  BasicBlock* loop_preheader_;

  // The block containing the backedge to the loop header.
  BasicBlock* loop_latch_;

  // A parent of a loop is the loop which contains it as a nested child loop.
  Loop* parent_;

  // Nested child loops of this loop.
  ChildrenList nested_loops_;

  // A set of all the basic blocks which comprise the loop structure. Will be
  // computed only when needed on demand.
  BasicBlockListTy loop_basic_blocks_;

  // Check that |bb| is inside the loop using domination property.
  // Note: this is for assertion purposes only, IsInsideLoop should be used
  // instead.
  bool IsBasicBlockInLoopSlow(const BasicBlock* bb);

  // Returns the loop preheader if it exists, returns nullptr otherwise.
  BasicBlock* FindLoopPreheader(DominatorAnalysis* dom_analysis);

  // Sets |latch| as the loop unique latch block. No checks are performed
  // here.
  inline void SetLatchBlockImpl(BasicBlock* latch) { loop_latch_ = latch; }
  // Sets |merge| as the loop merge block. No checks are performed here.
  inline void SetMergeBlockImpl(BasicBlock* merge) { loop_merge_ = merge; }

  // Each different loop |condition| affects how we calculate the number of
  // iterations using the |condition_value|, |init_value|, and |step_values| of
  // the induction variable. This method will return the number of iterations in
  // a loop with those values for a given |condition|.  Returns 0 if the number
  // of iterations could not be computed.
  int64_t GetIterations(spv::Op condition, int64_t condition_value,
                        int64_t init_value, int64_t step_value) const;

  // This is to allow for loops to be removed mid iteration without invalidating
  // the iterators.
  bool loop_is_marked_for_removal_;

  // This is only to allow LoopDescriptor::placeholder_top_loop_ to add top
  // level loops as child.
  friend class LoopDescriptor;
  friend class LoopUtils;
};

// Loop descriptions class for a given function.
// For a given function, the class builds loop nests information.
// The analysis expects a structured control flow.
class LoopDescriptor {
 public:
  // Iterator interface (depth first postorder traversal).
  using iterator = PostOrderTreeDFIterator<Loop>;
  using const_iterator = PostOrderTreeDFIterator<const Loop>;

  using pre_iterator = TreeDFIterator<Loop>;
  using const_pre_iterator = TreeDFIterator<const Loop>;

  // Creates a loop object for all loops found in |f|.
  LoopDescriptor(IRContext* context, const Function* f);

  // Disable copy constructor, to avoid double-free on destruction.
  LoopDescriptor(const LoopDescriptor&) = delete;
  // Move constructor.
  LoopDescriptor(LoopDescriptor&& other) : placeholder_top_loop_(nullptr) {
    // We need to take ownership of the Loop objects in the other
    // LoopDescriptor, to avoid double-free.
    loops_ = std::move(other.loops_);
    other.loops_.clear();
    basic_block_to_loop_ = std::move(other.basic_block_to_loop_);
    other.basic_block_to_loop_.clear();
    placeholder_top_loop_ = std::move(other.placeholder_top_loop_);
  }

  // Destructor
  ~LoopDescriptor();

  // Returns the number of loops found in the function.
  inline size_t NumLoops() const { return loops_.size(); }

  // Returns the loop at a particular |index|. The |index| must be in bounds,
  // check with NumLoops before calling.
  inline Loop& GetLoopByIndex(size_t index) const {
    assert(loops_.size() > index &&
           "Index out of range (larger than loop count)");
    return *loops_[index];
  }

  // Returns the loops in |this| in the order their headers appear in the
  // binary.
  std::vector<Loop*> GetLoopsInBinaryLayoutOrder();

  // Returns the inner most loop that contains the basic block id |block_id|.
  inline Loop* operator[](uint32_t block_id) const {
    return FindLoopForBasicBlock(block_id);
  }

  // Returns the inner most loop that contains the basic block |bb|.
  inline Loop* operator[](const BasicBlock* bb) const {
    return (*this)[bb->id()];
  }

  // Iterators for post order depth first traversal of the loops.
  // Inner most loops will be visited first.
  inline iterator begin() { return iterator::begin(&placeholder_top_loop_); }
  inline iterator end() { return iterator::end(&placeholder_top_loop_); }
  inline const_iterator begin() const { return cbegin(); }
  inline const_iterator end() const { return cend(); }
  inline const_iterator cbegin() const {
    return const_iterator::begin(&placeholder_top_loop_);
  }
  inline const_iterator cend() const {
    return const_iterator::end(&placeholder_top_loop_);
  }

  // Iterators for pre-order depth first traversal of the loops.
  // Inner most loops will be visited first.
  inline pre_iterator pre_begin() {
    return ++pre_iterator(&placeholder_top_loop_);
  }
  inline pre_iterator pre_end() { return pre_iterator(); }
  inline const_pre_iterator pre_begin() const { return pre_cbegin(); }
  inline const_pre_iterator pre_end() const { return pre_cend(); }
  inline const_pre_iterator pre_cbegin() const {
    return ++const_pre_iterator(&placeholder_top_loop_);
  }
  inline const_pre_iterator pre_cend() const { return const_pre_iterator(); }

  // Returns the inner most loop that contains the basic block |bb|.
  inline void SetBasicBlockToLoop(uint32_t bb_id, Loop* loop) {
    basic_block_to_loop_[bb_id] = loop;
  }

  // Mark the loop |loop_to_add| as needing to be added when the user calls
  // PostModificationCleanup. |parent| may be null.
  inline void AddLoop(std::unique_ptr<Loop>&& loop_to_add, Loop* parent) {
    loops_to_add_.emplace_back(std::make_pair(parent, std::move(loop_to_add)));
  }

  // Checks all loops in |this| and will create pre-headers for all loops
  // that don't have one. Returns |true| if any blocks were created.
  bool CreatePreHeaderBlocksIfMissing();

  // Should be called to preserve the LoopAnalysis after loops have been marked
  // for addition with AddLoop or MarkLoopForRemoval.
  void PostModificationCleanup();

  // Removes the basic block id |bb_id| from the block to loop mapping.
  inline void ForgetBasicBlock(uint32_t bb_id) {
    basic_block_to_loop_.erase(bb_id);
  }

  // Adds the loop |new_loop| and all its nested loops to the descriptor set.
  // The object takes ownership of all the loops.
  Loop* AddLoopNest(std::unique_ptr<Loop> new_loop);

  // Remove the loop |loop|.
  void RemoveLoop(Loop* loop);

  void SetAsTopLoop(Loop* loop) {
    assert(std::find(placeholder_top_loop_.begin(), placeholder_top_loop_.end(),
                     loop) == placeholder_top_loop_.end() &&
           "already registered");
    placeholder_top_loop_.nested_loops_.push_back(loop);
  }

  Loop* GetPlaceholderRootLoop() { return &placeholder_top_loop_; }
  const Loop* GetPlaceholderRootLoop() const { return &placeholder_top_loop_; }

 private:
  // TODO(dneto): This should be a vector of unique_ptr.  But VisualStudio 2013
  // is unable to compile it.
  using LoopContainerType = std::vector<Loop*>;

  using LoopsToAddContainerType =
      std::vector<std::pair<Loop*, std::unique_ptr<Loop>>>;

  // Creates loop descriptors for the function |f|.
  void PopulateList(IRContext* context, const Function* f);

  // Returns the inner most loop that contains the basic block id |block_id|.
  inline Loop* FindLoopForBasicBlock(uint32_t block_id) const {
    std::unordered_map<uint32_t, Loop*>::const_iterator it =
        basic_block_to_loop_.find(block_id);
    return it != basic_block_to_loop_.end() ? it->second : nullptr;
  }

  // Erase all the loop information.
  void ClearLoops();

  // A list of all the loops in the function.  This variable owns the Loop
  // objects.
  LoopContainerType loops_;

  // Placeholder root: this "loop" is only there to help iterators creation.
  Loop placeholder_top_loop_;

  std::unordered_map<uint32_t, Loop*> basic_block_to_loop_;

  // List of the loops marked for addition when PostModificationCleanup is
  // called.
  LoopsToAddContainerType loops_to_add_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_LOOP_DESCRIPTOR_H_
