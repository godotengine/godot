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
#include <unordered_set>

#include "source/val/function.h"

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

Construct::ConstructBlockSet Construct::blocks(Function* function) const {
  auto header = entry_block();
  auto merge = exit_block();
  assert(header);
  int header_depth = function->GetBlockDepth(const_cast<BasicBlock*>(header));
  ConstructBlockSet construct_blocks;
  std::unordered_set<BasicBlock*> corresponding_headers;
  for (auto& other : corresponding_constructs()) {
    corresponding_headers.insert(other->entry_block());
  }
  std::vector<BasicBlock*> stack;
  stack.push_back(const_cast<BasicBlock*>(header));
  while (!stack.empty()) {
    BasicBlock* block = stack.back();
    stack.pop_back();

    if (merge == block && ExitBlockIsMergeBlock()) {
      // Merge block is not part of the construct.
      continue;
    }

    if (corresponding_headers.count(block)) {
      // Entered a corresponding construct.
      continue;
    }

    int block_depth = function->GetBlockDepth(block);
    if (block_depth < header_depth) {
      // Broke to outer construct.
      continue;
    }

    // In a loop, the continue target is at a depth of the loop construct + 1.
    // A selection construct nested directly within the loop construct is also
    // at the same depth. It is valid, however, to branch directly to the
    // continue target from within the selection construct.
    if (block_depth == header_depth && type() == ConstructType::kSelection &&
        block->is_type(kBlockTypeContinue)) {
      // Continued to outer construct.
      continue;
    }

    if (!construct_blocks.insert(block).second) continue;

    if (merge != block) {
      for (auto succ : *block->successors()) {
        // All blocks in the construct must be dominated by the header.
        if (header->dominates(*succ)) {
          stack.push_back(succ);
        }
      }
    }
  }

  return construct_blocks;
}

}  // namespace val
}  // namespace spvtools
