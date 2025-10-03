// Copyright (c) 2016 Google Inc.
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

// This file defines the language constructs for representing a SPIR-V
// module in memory.

#ifndef SOURCE_OPT_BASIC_BLOCK_H_
#define SOURCE_OPT_BASIC_BLOCK_H_

#include <functional>
#include <iterator>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "source/opt/instruction.h"
#include "source/opt/instruction_list.h"
#include "source/opt/iterator.h"

namespace spvtools {
namespace opt {

class Function;
class IRContext;

// A SPIR-V basic block.
class BasicBlock {
 public:
  using iterator = InstructionList::iterator;
  using const_iterator = InstructionList::const_iterator;
  using reverse_iterator = std::reverse_iterator<InstructionList::iterator>;
  using const_reverse_iterator =
      std::reverse_iterator<InstructionList::const_iterator>;

  // Creates a basic block with the given starting |label|.
  inline explicit BasicBlock(std::unique_ptr<Instruction> label);

  explicit BasicBlock(const BasicBlock& bb) = delete;

  // Creates a clone of the basic block in the given |context|
  //
  // The parent function will default to null and needs to be explicitly set by
  // the user.
  //
  // If the inst-to-block map in |context| is valid, then the new instructions
  // will be inserted into the map.
  BasicBlock* Clone(IRContext*) const;

  // Sets the enclosing function for this basic block.
  void SetParent(Function* function) { function_ = function; }

  // Return the enclosing function
  inline Function* GetParent() const { return function_; }

  // Appends an instruction to this basic block.
  inline void AddInstruction(std::unique_ptr<Instruction> i);

  // Appends all of block's instructions (except label) to this block
  inline void AddInstructions(BasicBlock* bp);

  // The pointer to the label starting this basic block.
  std::unique_ptr<Instruction>& GetLabel() { return label_; }

  // The label starting this basic block.
  Instruction* GetLabelInst() { return label_.get(); }
  const Instruction* GetLabelInst() const { return label_.get(); }

  // Returns the merge instruction in this basic block, if it exists.
  // Otherwise return null.  May be used whenever tail() can be used.
  const Instruction* GetMergeInst() const;
  Instruction* GetMergeInst();

  // Returns the OpLoopMerge instruction in this basic block, if it exists.
  // Otherwise return null.  May be used whenever tail() can be used.
  const Instruction* GetLoopMergeInst() const;
  Instruction* GetLoopMergeInst();

  // Returns the id of the label at the top of this block
  inline uint32_t id() const { return label_->result_id(); }

  iterator begin() { return insts_.begin(); }
  iterator end() { return insts_.end(); }
  const_iterator begin() const { return insts_.cbegin(); }
  const_iterator end() const { return insts_.cend(); }
  const_iterator cbegin() const { return insts_.cbegin(); }
  const_iterator cend() const { return insts_.cend(); }

  reverse_iterator rbegin() { return reverse_iterator(end()); }
  reverse_iterator rend() { return reverse_iterator(begin()); }
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(cend());
  }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(cbegin());
  }
  const_reverse_iterator crbegin() const {
    return const_reverse_iterator(cend());
  }
  const_reverse_iterator crend() const {
    return const_reverse_iterator(cbegin());
  }

  // Returns an iterator pointing to the last instruction.  This may only
  // be used if this block has an instruction other than the OpLabel
  // that defines it.
  iterator tail() {
    assert(!insts_.empty());
    return --end();
  }

  // Returns a const iterator, but othewrise similar to tail().
  const_iterator ctail() const {
    assert(!insts_.empty());
    return --insts_.cend();
  }

  // Returns true if the basic block has at least one successor.
  inline bool hasSuccessor() const { return ctail()->IsBranch(); }

  // Runs the given function |f| on each instruction in this basic block, and
  // optionally on the debug line instructions that might precede them.
  inline void ForEachInst(const std::function<void(Instruction*)>& f,
                          bool run_on_debug_line_insts = false);
  inline void ForEachInst(const std::function<void(const Instruction*)>& f,
                          bool run_on_debug_line_insts = false) const;

  // Runs the given function |f| on each instruction in this basic block, and
  // optionally on the debug line instructions that might precede them. If |f|
  // returns false, iteration is terminated and this function returns false.
  inline bool WhileEachInst(const std::function<bool(Instruction*)>& f,
                            bool run_on_debug_line_insts = false);
  inline bool WhileEachInst(const std::function<bool(const Instruction*)>& f,
                            bool run_on_debug_line_insts = false) const;

  // Runs the given function |f| on each Phi instruction in this basic block,
  // and optionally on the debug line instructions that might precede them.
  inline void ForEachPhiInst(const std::function<void(Instruction*)>& f,
                             bool run_on_debug_line_insts = false);

  // Runs the given function |f| on each Phi instruction in this basic block,
  // and optionally on the debug line instructions that might precede them. If
  // |f| returns false, iteration is terminated and this function return false.
  inline bool WhileEachPhiInst(const std::function<bool(Instruction*)>& f,
                               bool run_on_debug_line_insts = false);

  // Runs the given function |f| on each label id of each successor block
  void ForEachSuccessorLabel(
      const std::function<void(const uint32_t)>& f) const;

  // Runs the given function |f| on each label id of each successor block.  If
  // |f| returns false, iteration is terminated and this function returns false.
  bool WhileEachSuccessorLabel(
      const std::function<bool(const uint32_t)>& f) const;

  // Runs the given function |f| on each label id of each successor block.
  // Modifying the pointed value will change the branch taken by the basic
  // block. It is the caller responsibility to update or invalidate the CFG.
  void ForEachSuccessorLabel(const std::function<void(uint32_t*)>& f);

  // Returns true if |block| is a direct successor of |this|.
  bool IsSuccessor(const BasicBlock* block) const;

  // Runs the given function |f| on the merge and continue label, if any
  void ForMergeAndContinueLabel(const std::function<void(const uint32_t)>& f);

  // Returns true if this basic block has any Phi instructions.
  bool HasPhiInstructions() {
    return !WhileEachPhiInst([](Instruction*) { return false; });
  }

  // Return true if this block is a loop header block.
  bool IsLoopHeader() const { return GetLoopMergeInst() != nullptr; }

  // Returns the ID of the merge block declared by a merge instruction in this
  // block, if any.  If none, returns zero.
  uint32_t MergeBlockIdIfAny() const;

  // Returns MergeBlockIdIfAny() and asserts that it is non-zero.
  uint32_t MergeBlockId() const;

  // Returns the ID of the continue block declared by a merge instruction in
  // this block, if any.  If none, returns zero.
  uint32_t ContinueBlockIdIfAny() const;

  // Returns ContinueBlockIdIfAny() and asserts that it is non-zero.
  uint32_t ContinueBlockId() const;

  // Returns the terminator instruction.  Assumes the terminator exists.
  Instruction* terminator() { return &*tail(); }
  const Instruction* terminator() const { return &*ctail(); }

  // Returns true if this basic block exits this function and returns to its
  // caller.
  bool IsReturn() const { return ctail()->IsReturn(); }

  // Returns true if this basic block exits this function or aborts execution.
  bool IsReturnOrAbort() const { return ctail()->IsReturnOrAbort(); }

  // Kill all instructions in this block. Whether or not to kill the label is
  // indicated by |killLabel|.
  void KillAllInsts(bool killLabel);

  // Splits this basic block into two. Returns a new basic block with label
  // |label_id| containing the instructions from |iter| onwards. Instructions
  // prior to |iter| remain in this basic block.  The new block will be added
  // to the function immediately after the original block.
  BasicBlock* SplitBasicBlock(IRContext* context, uint32_t label_id,
                              iterator iter);

  // Pretty-prints this basic block into a std::string by printing every
  // instruction in it.
  //
  // |options| are the disassembly options. SPV_BINARY_TO_TEXT_OPTION_NO_HEADER
  // is always added to |options|.
  std::string PrettyPrint(uint32_t options = 0u) const;

  // Dump this basic block on stderr.  Useful when running interactive
  // debuggers.
  void Dump() const;

 private:
  // The enclosing function.
  Function* function_;
  // The label starting this basic block.
  std::unique_ptr<Instruction> label_;
  // Instructions inside this basic block, but not the OpLabel.
  InstructionList insts_;
};

// Pretty-prints |block| to |str|. Returns |str|.
std::ostream& operator<<(std::ostream& str, const BasicBlock& block);

inline BasicBlock::BasicBlock(std::unique_ptr<Instruction> label)
    : function_(nullptr), label_(std::move(label)) {}

inline void BasicBlock::AddInstruction(std::unique_ptr<Instruction> i) {
  insts_.push_back(std::move(i));
}

inline void BasicBlock::AddInstructions(BasicBlock* bp) {
  auto bEnd = end();
  (void)bEnd.MoveBefore(&bp->insts_);
}

inline bool BasicBlock::WhileEachInst(
    const std::function<bool(Instruction*)>& f, bool run_on_debug_line_insts) {
  if (label_) {
    if (!label_->WhileEachInst(f, run_on_debug_line_insts)) return false;
  }
  if (insts_.empty()) {
    return true;
  }

  Instruction* inst = &insts_.front();
  while (inst != nullptr) {
    Instruction* next_instruction = inst->NextNode();
    if (!inst->WhileEachInst(f, run_on_debug_line_insts)) return false;
    inst = next_instruction;
  }
  return true;
}

inline bool BasicBlock::WhileEachInst(
    const std::function<bool(const Instruction*)>& f,
    bool run_on_debug_line_insts) const {
  if (label_) {
    if (!static_cast<const Instruction*>(label_.get())
             ->WhileEachInst(f, run_on_debug_line_insts))
      return false;
  }
  for (const auto& inst : insts_) {
    if (!static_cast<const Instruction*>(&inst)->WhileEachInst(
            f, run_on_debug_line_insts))
      return false;
  }
  return true;
}

inline void BasicBlock::ForEachInst(const std::function<void(Instruction*)>& f,
                                    bool run_on_debug_line_insts) {
  WhileEachInst(
      [&f](Instruction* inst) {
        f(inst);
        return true;
      },
      run_on_debug_line_insts);
}

inline void BasicBlock::ForEachInst(
    const std::function<void(const Instruction*)>& f,
    bool run_on_debug_line_insts) const {
  WhileEachInst(
      [&f](const Instruction* inst) {
        f(inst);
        return true;
      },
      run_on_debug_line_insts);
}

inline bool BasicBlock::WhileEachPhiInst(
    const std::function<bool(Instruction*)>& f, bool run_on_debug_line_insts) {
  if (insts_.empty()) {
    return true;
  }

  Instruction* inst = &insts_.front();
  while (inst != nullptr) {
    Instruction* next_instruction = inst->NextNode();
    if (inst->opcode() != spv::Op::OpPhi) break;
    if (!inst->WhileEachInst(f, run_on_debug_line_insts)) return false;
    inst = next_instruction;
  }
  return true;
}

inline void BasicBlock::ForEachPhiInst(
    const std::function<void(Instruction*)>& f, bool run_on_debug_line_insts) {
  WhileEachPhiInst(
      [&f](Instruction* inst) {
        f(inst);
        return true;
      },
      run_on_debug_line_insts);
}

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_BASIC_BLOCK_H_
