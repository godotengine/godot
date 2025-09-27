
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

#ifndef SOURCE_OPT_INSTRUCTION_LIST_H_
#define SOURCE_OPT_INSTRUCTION_LIST_H_

#include <cassert>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "source/latest_version_spirv_header.h"
#include "source/operand.h"
#include "source/opt/instruction.h"
#include "source/util/ilist.h"
#include "spirv-tools/libspirv.h"

namespace spvtools {
namespace opt {

// This class is intended to be the container for Instructions.  This container
// owns the instructions that are in it.  When removing an Instruction from the
// list, the caller is assuming responsibility for deleting the storage.
//
// TODO: Because there are a number of other data structures that will want
// pointers to instruction, ownership should probably be moved to the module.
// Because of that I have not made the ownership passing in this class fully
// explicit.  For example, RemoveFromList takes ownership from the list, but
// does not return an std::unique_ptr to signal that.  When we fully decide on
// ownership, this will have to be fixed up one way or the other.
class InstructionList : public utils::IntrusiveList<Instruction> {
 public:
  InstructionList() = default;
  InstructionList(InstructionList&& that)
      : utils::IntrusiveList<Instruction>(std::move(that)) {}
  InstructionList& operator=(InstructionList&& that) {
    auto p = static_cast<utils::IntrusiveList<Instruction>*>(this);
    *p = std::move(that);
    return *this;
  }

  // Destroy this list and any instructions in the list.
  inline ~InstructionList() override;

  class iterator : public utils::IntrusiveList<Instruction>::iterator {
   public:
    iterator(const utils::IntrusiveList<Instruction>::iterator& i)
        : utils::IntrusiveList<Instruction>::iterator(i) {}
    iterator(Instruction* i) : utils::IntrusiveList<Instruction>::iterator(i) {}

    iterator& operator++() {
      utils::IntrusiveList<Instruction>::iterator::operator++();
      return *this;
    }

    iterator& operator--() {
      utils::IntrusiveList<Instruction>::iterator::operator--();
      return *this;
    }

    // DEPRECATED: Please use MoveBefore with an InstructionList instead.
    //
    // Moves the nodes in |list| to the list that |this| points to.  The
    // positions of the nodes will be immediately before the element pointed to
    // by the iterator.  The return value will be an iterator pointing to the
    // first of the newly inserted elements.  Ownership of the elements in
    // |list| is now passed on to |*this|.
    iterator InsertBefore(std::vector<std::unique_ptr<Instruction>>&& list);

    // The node |i| will be inserted immediately before |this|. The return value
    // will be an iterator pointing to the newly inserted node.  The owner of
    // |*i| becomes |*this|
    iterator InsertBefore(std::unique_ptr<Instruction>&& i);

    // Removes the node from the list, and deletes the storage.  Returns a valid
    // iterator to the next node.
    iterator Erase() {
      iterator_template next_node = *this;
      ++next_node;
      node_->RemoveFromList();
      delete node_;
      return next_node;
    }
  };

  iterator begin() { return utils::IntrusiveList<Instruction>::begin(); }
  iterator end() { return utils::IntrusiveList<Instruction>::end(); }
  const_iterator begin() const {
    return utils::IntrusiveList<Instruction>::begin();
  }
  const_iterator end() const {
    return utils::IntrusiveList<Instruction>::end();
  }

  void push_back(std::unique_ptr<Instruction>&& inst) {
    utils::IntrusiveList<Instruction>::push_back(inst.release());
  }

  // Same as in the base class, except it will delete the data as well.
  inline void clear();

  // Runs the given function |f| on the instructions in the list and optionally
  // on the preceding debug line instructions.
  inline void ForEachInst(const std::function<void(Instruction*)>& f,
                          bool run_on_debug_line_insts) {
    auto next = begin();
    for (auto i = next; i != end(); i = next) {
      ++next;
      i->ForEachInst(f, run_on_debug_line_insts);
    }
  }
};

InstructionList::~InstructionList() { clear(); }

void InstructionList::clear() {
  while (!empty()) {
    Instruction* inst = &front();
    inst->RemoveFromList();
    delete inst;
  }
}

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_INSTRUCTION_LIST_H_
