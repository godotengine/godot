// Copyright (c) 2015-2016 The Khronos Group Inc.
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

#include "source/val/construct.h"

#include <cassert>
#include <cstddef>

#include "source/val/function.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {

Construct::Construct(ConstructType construct_type, BasicBlock* entry,
                     BasicBlock* exit, std::vector<Construct*> constructs)
    : type_(construct_type),
      corresponding_constructs_(constructs),
      entry_block_(entry),
      exit_block_(exit) {}

ConstructType Construct::type() const { return type_; }

const std::vector<Construct*>& Construct::corresponding_constructs() const {
  return corresponding_constructs_;
}
std::vector<Construct*>& Construct::corresponding_constructs() {
  return corresponding_constructs_;
}

bool ValidateConstructSize(ConstructType type, size_t size) {
  switch (type) {
    case ConstructType::kSelection:
      return size == 0;
    case ConstructType::kContinue:
      return size == 1;
    case ConstructType::kLoop:
      return size == 1;
    case ConstructType::kCase:
      return size >= 1;
    default:
      assert(1 == 0 && "Type not defined");
  }
  return false;
}

void Construct::set_corresponding_constructs(
    std::vector<Construct*> constructs) {
  assert(ValidateConstructSize(type_, constructs.size()));
  corresponding_constructs_ = constructs;
}

const BasicBlock* Construct::entry_block() const { return entry_block_; }
BasicBlock* Construct::entry_block() { return entry_block_; }

const BasicBlock* Construct::exit_block() const { return exit_block_; }
BasicBlock* Construct::exit_block() { return exit_block_; }

void Construct::set_exit(BasicBlock* block) { exit_block_ = block; }

Construct::ConstructBlockSet Construct::blocks(Function* /*function*/) const {
  const auto header = entry_block();
  const auto exit = exit_block();
  const bool is_continue = type() == ConstructType::kContinue;
  const bool is_loop = type() == ConstructType::kLoop;
  const BasicBlock* continue_header = nullptr;
  if (is_loop) {
    // The only corresponding construct for a loop is the continue.
    continue_header = (*corresponding_constructs().begin())->entry_block();
  }
  std::vector<BasicBlock*> stack;
  stack.push_back(const_cast<BasicBlock*>(header));
  ConstructBlockSet construct_blocks;
  while (!stack.empty()) {
    auto* block = stack.back();
    stack.pop_back();

    if (header->structurally_dominates(*block)) {
      bool include = false;
      if (is_continue && exit->structurally_postdominates(*block)) {
        // Continue construct include blocks dominated by the continue target
        // and post-dominated by the back-edge block.
        include = true;
      } else if (!exit->structurally_dominates(*block)) {
        // Selection and loop constructs include blocks dominated by the header
        // and not dominated by the merge.
        include = true;
        if (is_loop && continue_header->structurally_dominates(*block)) {
          // Loop constructs have an additional constraint that they do not
          // include blocks dominated by the continue construct. Since all
          // blocks in the continue construct are dominated by the continue
          // target, we just test for dominance by continue target.
          include = false;
        }
      }
      if (include) {
        if (!construct_blocks.insert(block).second) continue;

        for (auto succ : *block->structural_successors()) {
          stack.push_back(succ);
        }
      }
    }
  }

  return construct_blocks;
}

bool Construct::IsStructuredExit(ValidationState_t& _, BasicBlock* dest) const {
  // Structured Exits:
  // - Selection:
  //  - branch to its merge
  //  - branch to nearest enclosing loop merge or continue
  //  - branch to nearest enclosing switch selection merge
  // - Loop:
  //  - branch to its merge
  //  - branch to its continue
  // - Continue:
  //  - branch to loop header
  //  - branch to loop merge
  //
  // Note: we will never see a case construct here.
  assert(type() != ConstructType::kCase);
  if (type() == ConstructType::kLoop) {
    auto header = entry_block();
    auto terminator = header->terminator();
    auto index = terminator - &_.ordered_instructions()[0];
    auto merge_inst = &_.ordered_instructions()[index - 1];
    auto merge_block_id = merge_inst->GetOperandAs<uint32_t>(0u);
    auto continue_block_id = merge_inst->GetOperandAs<uint32_t>(1u);
    if (dest->id() == merge_block_id || dest->id() == continue_block_id) {
      return true;
    }
  } else if (type() == ConstructType::kContinue) {
    auto loop_construct = corresponding_constructs()[0];
    auto header = loop_construct->entry_block();
    auto terminator = header->terminator();
    auto index = terminator - &_.ordered_instructions()[0];
    auto merge_inst = &_.ordered_instructions()[index - 1];
    auto merge_block_id = merge_inst->GetOperandAs<uint32_t>(0u);
    if (dest == header || dest->id() == merge_block_id) {
      return true;
    }
  } else {
    assert(type() == ConstructType::kSelection);
    if (dest == exit_block()) {
      return true;
    }

    // The next block in the traversal is either:
    //  i.  The header block that declares |block| as its merge block.
    //  ii. The immediate dominator of |block|.
    auto NextBlock = [](const BasicBlock* block) -> const BasicBlock* {
      for (auto& use : block->label()->uses()) {
        if ((use.first->opcode() == spv::Op::OpLoopMerge ||
             use.first->opcode() == spv::Op::OpSelectionMerge) &&
            use.second == 1 &&
            use.first->block()->structurally_dominates(*block) &&
            // A header likely declared itself as its merge.
            use.first->block() != block) {
          return use.first->block();
        }
      }
      return block->immediate_structural_dominator();
    };

    bool seen_switch = false;
    auto header = entry_block();
    auto block = NextBlock(header);
    while (block) {
      auto terminator = block->terminator();
      auto index = terminator - &_.ordered_instructions()[0];
      auto merge_inst = &_.ordered_instructions()[index - 1];
      if (merge_inst->opcode() == spv::Op::OpLoopMerge ||
          (header->terminator()->opcode() != spv::Op::OpSwitch &&
           merge_inst->opcode() == spv::Op::OpSelectionMerge &&
           terminator->opcode() == spv::Op::OpSwitch)) {
        auto merge_target = merge_inst->GetOperandAs<uint32_t>(0u);
        auto merge_block = merge_inst->function()->GetBlock(merge_target).first;
        if (merge_block->structurally_dominates(*header)) {
          block = NextBlock(block);
          continue;
        }

        if ((!seen_switch || merge_inst->opcode() == spv::Op::OpLoopMerge) &&
            dest->id() == merge_target) {
          return true;
        } else if (merge_inst->opcode() == spv::Op::OpLoopMerge) {
          auto continue_target = merge_inst->GetOperandAs<uint32_t>(1u);
          if (dest->id() == continue_target) {
            return true;
          }
        }

        if (terminator->opcode() == spv::Op::OpSwitch) {
          seen_switch = true;
        }

        // Hit an enclosing loop and didn't break or continue.
        if (merge_inst->opcode() == spv::Op::OpLoopMerge) return false;
      }

      block = NextBlock(block);
    }
  }

  return false;
}

}  // namespace val
}  // namespace spvtools
