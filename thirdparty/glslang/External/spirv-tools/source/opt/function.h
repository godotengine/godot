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

#ifndef SOURCE_OPT_FUNCTION_H_
#define SOURCE_OPT_FUNCTION_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "source/opt/basic_block.h"
#include "source/opt/instruction.h"
#include "source/opt/iterator.h"

namespace spvtools {
namespace opt {

class CFG;
class IRContext;
class Module;

// A SPIR-V function.
class Function {
 public:
  using iterator = UptrVectorIterator<BasicBlock>;
  using const_iterator = UptrVectorIterator<BasicBlock, true>;

  // Creates a function instance declared by the given OpFunction instruction
  // |def_inst|.
  inline explicit Function(std::unique_ptr<Instruction> def_inst);

  explicit Function(const Function& f) = delete;

  // Creates a clone of the instruction in the given |context|
  //
  // The parent module will default to null and needs to be explicitly set by
  // the user.
  Function* Clone(IRContext*) const;
  // The OpFunction instruction that begins the definition of this function.
  Instruction& DefInst() { return *def_inst_; }
  const Instruction& DefInst() const { return *def_inst_; }

  // Appends a parameter to this function.
  inline void AddParameter(std::unique_ptr<Instruction> p);
  // Appends a debug instruction in function header to this function.
  inline void AddDebugInstructionInHeader(std::unique_ptr<Instruction> p);
  // Appends a basic block to this function.
  inline void AddBasicBlock(std::unique_ptr<BasicBlock> b);
  // Appends a basic block to this function at the position |ip|.
  inline void AddBasicBlock(std::unique_ptr<BasicBlock> b, iterator ip);
  template <typename T>
  inline void AddBasicBlocks(T begin, T end, iterator ip);

  // Move basic block with |id| to the position after |ip|. Both have to be
  // contained in this function.
  inline void MoveBasicBlockToAfter(uint32_t id, BasicBlock* ip);

  // Delete all basic blocks that contain no instructions.
  inline void RemoveEmptyBlocks();

  // Removes a parameter from the function with result id equal to |id|.
  // Does nothing if the function doesn't have such a parameter.
  inline void RemoveParameter(uint32_t id);

  // Saves the given function end instruction.
  inline void SetFunctionEnd(std::unique_ptr<Instruction> end_inst);

  // Add a non-semantic instruction that succeeds this function in the module.
  // These instructions are maintained in the order they are added.
  inline void AddNonSemanticInstruction(
      std::unique_ptr<Instruction> non_semantic);

  // Returns the given function end instruction.
  inline Instruction* EndInst() { return end_inst_.get(); }
  inline const Instruction* EndInst() const { return end_inst_.get(); }

  // Returns function's id
  inline uint32_t result_id() const { return def_inst_->result_id(); }

  // Returns function's return type id
  inline uint32_t type_id() const { return def_inst_->type_id(); }

  // Returns the function's control mask
  inline uint32_t control_mask() const { return def_inst_->GetSingleWordInOperand(0); }

  // Returns the entry basic block for this function.
  const std::unique_ptr<BasicBlock>& entry() const { return blocks_.front(); }

  // Returns the last basic block in this function.
  BasicBlock* tail() { return blocks_.back().get(); }
  const BasicBlock* tail() const { return blocks_.back().get(); }

  iterator begin() { return iterator(&blocks_, blocks_.begin()); }
  iterator end() { return iterator(&blocks_, blocks_.end()); }
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }
  const_iterator cbegin() const {
    return const_iterator(&blocks_, blocks_.cbegin());
  }
  const_iterator cend() const {
    return const_iterator(&blocks_, blocks_.cend());
  }

  // Returns an iterator to the basic block |id|.
  iterator FindBlock(uint32_t bb_id) {
    return std::find_if(begin(), end(), [bb_id](const BasicBlock& it_bb) {
      return bb_id == it_bb.id();
    });
  }

  // Runs the given function |f| on instructions in this function, in order,
  // and optionally on debug line instructions that might precede them and
  // non-semantic instructions that succceed the function.
  void ForEachInst(const std::function<void(Instruction*)>& f,
                   bool run_on_debug_line_insts = false,
                   bool run_on_non_semantic_insts = false);
  void ForEachInst(const std::function<void(const Instruction*)>& f,
                   bool run_on_debug_line_insts = false,
                   bool run_on_non_semantic_insts = false) const;
  // Runs the given function |f| on instructions in this function, in order,
  // and optionally on debug line instructions that might precede them and
  // non-semantic instructions that succeed the function.  If |f| returns
  // false, iteration is terminated and this function returns false.
  bool WhileEachInst(const std::function<bool(Instruction*)>& f,
                     bool run_on_debug_line_insts = false,
                     bool run_on_non_semantic_insts = false);
  bool WhileEachInst(const std::function<bool(const Instruction*)>& f,
                     bool run_on_debug_line_insts = false,
                     bool run_on_non_semantic_insts = false) const;

  // Runs the given function |f| on each parameter instruction in this function,
  // in order, and optionally on debug line instructions that might precede
  // them.
  void ForEachParam(const std::function<void(const Instruction*)>& f,
                    bool run_on_debug_line_insts = false) const;
  void ForEachParam(const std::function<void(Instruction*)>& f,
                    bool run_on_debug_line_insts = false);

  // Runs the given function |f| on each debug instruction in this function's
  // header in order.
  void ForEachDebugInstructionsInHeader(
      const std::function<void(Instruction*)>& f);

  BasicBlock* InsertBasicBlockAfter(std::unique_ptr<BasicBlock>&& new_block,
                                    BasicBlock* position);

  BasicBlock* InsertBasicBlockBefore(std::unique_ptr<BasicBlock>&& new_block,
                                     BasicBlock* position);

  // Returns true if the function has a return block other than the exit block.
  bool HasEarlyReturn() const;

  // Returns true if the function calls itself either directly or indirectly.
  bool IsRecursive() const;

  // Pretty-prints all the basic blocks in this function into a std::string.
  //
  // |options| are the disassembly options. SPV_BINARY_TO_TEXT_OPTION_NO_HEADER
  // is always added to |options|.
  std::string PrettyPrint(uint32_t options = 0u) const;

  // Dump this function on stderr.  Useful when running interactive
  // debuggers.
  void Dump() const;

  // Returns true is a function declaration and not a function definition.
  bool IsDeclaration() { return begin() == end(); }

  // Reorders the basic blocks in the function to match the structured order.
  void ReorderBasicBlocksInStructuredOrder();

 private:
  // Reorders the basic blocks in the function to match the order given by the
  // range |{begin,end}|.  The range must contain every basic block in the
  // function, and no extras.
  template <class It>
  void ReorderBasicBlocks(It begin, It end);

  template <class It>
  bool ContainsAllBlocksInTheFunction(It begin, It end);

  // The OpFunction instruction that begins the definition of this function.
  std::unique_ptr<Instruction> def_inst_;
  // All parameters to this function.
  std::vector<std::unique_ptr<Instruction>> params_;
  // All debug instructions in this function's header.
  InstructionList debug_insts_in_header_;
  // All basic blocks inside this function in specification order
  std::vector<std::unique_ptr<BasicBlock>> blocks_;
  // The OpFunctionEnd instruction.
  std::unique_ptr<Instruction> end_inst_;
  // Non-semantic instructions succeeded by this function.
  std::vector<std::unique_ptr<Instruction>> non_semantic_;
};

// Pretty-prints |func| to |str|. Returns |str|.
std::ostream& operator<<(std::ostream& str, const Function& func);

inline Function::Function(std::unique_ptr<Instruction> def_inst)
    : def_inst_(std::move(def_inst)), end_inst_() {}

inline void Function::AddParameter(std::unique_ptr<Instruction> p) {
  params_.emplace_back(std::move(p));
}

inline void Function::AddDebugInstructionInHeader(
    std::unique_ptr<Instruction> p) {
  debug_insts_in_header_.push_back(std::move(p));
}

inline void Function::AddBasicBlock(std::unique_ptr<BasicBlock> b) {
  AddBasicBlock(std::move(b), end());
}

inline void Function::AddBasicBlock(std::unique_ptr<BasicBlock> b,
                                    iterator ip) {
  b->SetParent(this);
  ip.InsertBefore(std::move(b));
}

template <typename T>
inline void Function::AddBasicBlocks(T src_begin, T src_end, iterator ip) {
  blocks_.insert(ip.Get(), std::make_move_iterator(src_begin),
                 std::make_move_iterator(src_end));
}

inline void Function::MoveBasicBlockToAfter(uint32_t id, BasicBlock* ip) {
  std::unique_ptr<BasicBlock> block_to_move = std::move(*FindBlock(id).Get());
  blocks_.erase(std::find(std::begin(blocks_), std::end(blocks_), nullptr));

  assert(block_to_move->GetParent() == ip->GetParent() &&
         "Both blocks have to be in the same function.");

  InsertBasicBlockAfter(std::move(block_to_move), ip);
}

inline void Function::RemoveEmptyBlocks() {
  auto first_empty =
      std::remove_if(std::begin(blocks_), std::end(blocks_),
                     [](const std::unique_ptr<BasicBlock>& bb) -> bool {
                       return bb->GetLabelInst()->opcode() == spv::Op::OpNop;
                     });
  blocks_.erase(first_empty, std::end(blocks_));
}

inline void Function::RemoveParameter(uint32_t id) {
  params_.erase(std::remove_if(params_.begin(), params_.end(),
                               [id](const std::unique_ptr<Instruction>& param) {
                                 return param->result_id() == id;
                               }),
                params_.end());
}

inline void Function::SetFunctionEnd(std::unique_ptr<Instruction> end_inst) {
  end_inst_ = std::move(end_inst);
}

inline void Function::AddNonSemanticInstruction(
    std::unique_ptr<Instruction> non_semantic) {
  non_semantic_.emplace_back(std::move(non_semantic));
}

template <class It>
void Function::ReorderBasicBlocks(It begin, It end) {
  // Asserts to make sure every node in the function is in new_order.
  assert(ContainsAllBlocksInTheFunction(begin, end));

  // We have a pointer to all the elements in order, so we can release all
  // pointers in |block_|, and then create the new unique pointers from |{begin,
  // end}|.
  std::for_each(blocks_.begin(), blocks_.end(),
                [](std::unique_ptr<BasicBlock>& bb) { bb.release(); });
  std::transform(begin, end, blocks_.begin(), [](BasicBlock* bb) {
    return std::unique_ptr<BasicBlock>(bb);
  });
}

template <class It>
bool Function::ContainsAllBlocksInTheFunction(It begin, It end) {
  std::unordered_multiset<BasicBlock*> range(begin, end);
  if (range.size() != blocks_.size()) {
    return false;
  }

  for (auto& bb : blocks_) {
    if (range.count(bb.get()) == 0) return false;
  }
  return true;
}

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_FUNCTION_H_
