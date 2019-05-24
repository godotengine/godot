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

  // Saves the given function end instruction.
  inline void SetFunctionEnd(std::unique_ptr<Instruction> end_inst);

  // Returns the given function end instruction.
  inline Instruction* EndInst() { return end_inst_.get(); }
  inline const Instruction* EndInst() const { return end_inst_.get(); }

  // Returns function's id
  inline uint32_t result_id() const { return def_inst_->result_id(); }

  // Returns function's return type id
  inline uint32_t type_id() const { return def_inst_->type_id(); }

  // Returns the entry basic block for this function.
  const std::unique_ptr<BasicBlock>& entry() const { return blocks_.front(); }

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

  // Runs the given function |f| on each instruction in this function, and
  // optionally on debug line instructions that might precede them.
  void ForEachInst(const std::function<void(Instruction*)>& f,
                   bool run_on_debug_line_insts = false);
  void ForEachInst(const std::function<void(const Instruction*)>& f,
                   bool run_on_debug_line_insts = false) const;
  bool WhileEachInst(const std::function<bool(Instruction*)>& f,
                     bool run_on_debug_line_insts = false);
  bool WhileEachInst(const std::function<bool(const Instruction*)>& f,
                     bool run_on_debug_line_insts = false) const;

  // Runs the given function |f| on each parameter instruction in this function,
  // and optionally on debug line instructions that might precede them.
  void ForEachParam(const std::function<void(const Instruction*)>& f,
                    bool run_on_debug_line_insts = false) const;
  void ForEachParam(const std::function<void(Instruction*)>& f,
                    bool run_on_debug_line_insts = false);

  BasicBlock* InsertBasicBlockAfter(std::unique_ptr<BasicBlock>&& new_block,
                                    BasicBlock* position);

  // Return true if the function calls itself either directly or indirectly.
  bool IsRecursive() const;

  // Pretty-prints all the basic blocks in this function into a std::string.
  //
  // |options| are the disassembly options. SPV_BINARY_TO_TEXT_OPTION_NO_HEADER
  // is always added to |options|.
  std::string PrettyPrint(uint32_t options = 0u) const;

  // Dump this function on stderr.  Useful when running interactive
  // debuggers.
  void Dump() const;

 private:
  // The OpFunction instruction that begins the definition of this function.
  std::unique_ptr<Instruction> def_inst_;
  // All parameters to this function.
  std::vector<std::unique_ptr<Instruction>> params_;
  // All basic blocks inside this function in specification order
  std::vector<std::unique_ptr<BasicBlock>> blocks_;
  // The OpFunctionEnd instruction.
  std::unique_ptr<Instruction> end_inst_;
};

// Pretty-prints |func| to |str|. Returns |str|.
std::ostream& operator<<(std::ostream& str, const Function& func);

inline Function::Function(std::unique_ptr<Instruction> def_inst)
    : def_inst_(std::move(def_inst)), end_inst_() {}

inline void Function::AddParameter(std::unique_ptr<Instruction> p) {
  params_.emplace_back(std::move(p));
}

inline void Function::AddBasicBlock(std::unique_ptr<BasicBlock> b) {
  AddBasicBlock(std::move(b), end());
}

inline void Function::AddBasicBlock(std::unique_ptr<BasicBlock> b,
                                    iterator ip) {
  ip.InsertBefore(std::move(b));
}

template <typename T>
inline void Function::AddBasicBlocks(T src_begin, T src_end, iterator ip) {
  blocks_.insert(ip.Get(), std::make_move_iterator(src_begin),
                 std::make_move_iterator(src_end));
}

inline void Function::MoveBasicBlockToAfter(uint32_t id, BasicBlock* ip) {
  auto block_to_move = std::move(*FindBlock(id).Get());

  assert(block_to_move->GetParent() == ip->GetParent() &&
         "Both blocks have to be in the same function.");

  InsertBasicBlockAfter(std::move(block_to_move), ip);
  blocks_.erase(std::find(std::begin(blocks_), std::end(blocks_), nullptr));
}

inline void Function::RemoveEmptyBlocks() {
  auto first_empty =
      std::remove_if(std::begin(blocks_), std::end(blocks_),
                     [](const std::unique_ptr<BasicBlock>& bb) -> bool {
                       return bb->GetLabelInst()->opcode() == SpvOpNop;
                     });
  blocks_.erase(first_empty, std::end(blocks_));
}

inline void Function::SetFunctionEnd(std::unique_ptr<Instruction> end_inst) {
  end_inst_ = std::move(end_inst);
}

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_FUNCTION_H_
