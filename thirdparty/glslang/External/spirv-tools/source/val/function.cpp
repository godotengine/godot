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

#include "source/val/function.h"

#include <algorithm>
#include <cassert>
#include <sstream>
#include <unordered_map>
#include <utility>

#include "source/cfa.h"
#include "source/val/basic_block.h"
#include "source/val/construct.h"
#include "source/val/validate.h"

namespace spvtools {
namespace val {

// Universal Limit of ResultID + 1
static const uint32_t kInvalidId = 0x400000;

Function::Function(uint32_t function_id, uint32_t result_type_id,
                   spv::FunctionControlMask function_control,
                   uint32_t function_type_id)
    : id_(function_id),
      function_type_id_(function_type_id),
      result_type_id_(result_type_id),
      function_control_(function_control),
      declaration_type_(FunctionDecl::kFunctionDeclUnknown),
      end_has_been_registered_(false),
      blocks_(),
      current_block_(nullptr),
      pseudo_entry_block_(0),
      pseudo_exit_block_(kInvalidId),
      cfg_constructs_(),
      variable_ids_(),
      parameter_ids_() {}

bool Function::IsFirstBlock(uint32_t block_id) const {
  return !ordered_blocks_.empty() && *first_block() == block_id;
}

spv_result_t Function::RegisterFunctionParameter(uint32_t parameter_id,
                                                 uint32_t type_id) {
  assert(current_block_ == nullptr &&
         "RegisterFunctionParameter can only be called when parsing the binary "
         "outside of a block");
  // TODO(umar): Validate function parameter type order and count
  // TODO(umar): Use these variables to validate parameter type
  (void)parameter_id;
  (void)type_id;
  return SPV_SUCCESS;
}

spv_result_t Function::RegisterLoopMerge(uint32_t merge_id,
                                         uint32_t continue_id) {
  RegisterBlock(merge_id, false);
  RegisterBlock(continue_id, false);
  BasicBlock& merge_block = blocks_.at(merge_id);
  BasicBlock& continue_target_block = blocks_.at(continue_id);
  assert(current_block_ &&
         "RegisterLoopMerge must be called when called within a block");
  current_block_->RegisterStructuralSuccessor(&merge_block);
  current_block_->RegisterStructuralSuccessor(&continue_target_block);

  current_block_->set_type(kBlockTypeLoop);
  merge_block.set_type(kBlockTypeMerge);
  continue_target_block.set_type(kBlockTypeContinue);
  Construct& loop_construct =
      AddConstruct({ConstructType::kLoop, current_block_, &merge_block});
  Construct& continue_construct =
      AddConstruct({ConstructType::kContinue, &continue_target_block});

  continue_construct.set_corresponding_constructs({&loop_construct});
  loop_construct.set_corresponding_constructs({&continue_construct});
  merge_block_header_[&merge_block] = current_block_;
  if (continue_target_headers_.find(&continue_target_block) ==
      continue_target_headers_.end()) {
    continue_target_headers_[&continue_target_block] = {current_block_};
  } else {
    continue_target_headers_[&continue_target_block].push_back(current_block_);
  }

  return SPV_SUCCESS;
}

spv_result_t Function::RegisterSelectionMerge(uint32_t merge_id) {
  RegisterBlock(merge_id, false);
  BasicBlock& merge_block = blocks_.at(merge_id);
  current_block_->set_type(kBlockTypeSelection);
  merge_block.set_type(kBlockTypeMerge);
  merge_block_header_[&merge_block] = current_block_;
  current_block_->RegisterStructuralSuccessor(&merge_block);

  AddConstruct({ConstructType::kSelection, current_block(), &merge_block});

  return SPV_SUCCESS;
}

spv_result_t Function::RegisterSetFunctionDeclType(FunctionDecl type) {
  assert(declaration_type_ == FunctionDecl::kFunctionDeclUnknown);
  declaration_type_ = type;
  return SPV_SUCCESS;
}

spv_result_t Function::RegisterBlock(uint32_t block_id, bool is_definition) {
  assert(
      declaration_type_ == FunctionDecl::kFunctionDeclDefinition &&
      "RegisterBlocks can only be called after declaration_type_ is defined");

  std::unordered_map<uint32_t, BasicBlock>::iterator inserted_block;
  bool success = false;
  tie(inserted_block, success) =
      blocks_.insert({block_id, BasicBlock(block_id)});
  if (is_definition) {  // new block definition
    assert(current_block_ == nullptr &&
           "Register Block can only be called when parsing a binary outside of "
           "a BasicBlock");

    undefined_blocks_.erase(block_id);
    current_block_ = &inserted_block->second;
    ordered_blocks_.push_back(current_block_);
  } else if (success) {  // Block doesn't exist but this is not a definition
    undefined_blocks_.insert(block_id);
  }

  return SPV_SUCCESS;
}

void Function::RegisterBlockEnd(std::vector<uint32_t> next_list) {
  assert(
      current_block_ &&
      "RegisterBlockEnd can only be called when parsing a binary in a block");
  std::vector<BasicBlock*> next_blocks;
  next_blocks.reserve(next_list.size());

  std::unordered_map<uint32_t, BasicBlock>::iterator inserted_block;
  bool success;
  for (uint32_t successor_id : next_list) {
    tie(inserted_block, success) =
        blocks_.insert({successor_id, BasicBlock(successor_id)});
    if (success) {
      undefined_blocks_.insert(successor_id);
    }
    next_blocks.push_back(&inserted_block->second);
  }

  if (current_block_->is_type(kBlockTypeLoop)) {
    // For each loop header, record the set of its successors, and include
    // its continue target if the continue target is not the loop header
    // itself.
    std::vector<BasicBlock*>& next_blocks_plus_continue_target =
        loop_header_successors_plus_continue_target_map_[current_block_];
    next_blocks_plus_continue_target = next_blocks;
    auto continue_target =
        FindConstructForEntryBlock(current_block_, ConstructType::kLoop)
            .corresponding_constructs()
            .back()
            ->entry_block();
    if (continue_target != current_block_) {
      next_blocks_plus_continue_target.push_back(continue_target);
    }
  }

  current_block_->RegisterSuccessors(next_blocks);
  current_block_ = nullptr;
  return;
}

void Function::RegisterFunctionEnd() {
  if (!end_has_been_registered_) {
    end_has_been_registered_ = true;

    ComputeAugmentedCFG();
  }
}

size_t Function::block_count() const { return blocks_.size(); }

size_t Function::undefined_block_count() const {
  return undefined_blocks_.size();
}

const std::vector<BasicBlock*>& Function::ordered_blocks() const {
  return ordered_blocks_;
}
std::vector<BasicBlock*>& Function::ordered_blocks() { return ordered_blocks_; }

const BasicBlock* Function::current_block() const { return current_block_; }
BasicBlock* Function::current_block() { return current_block_; }

const std::list<Construct>& Function::constructs() const {
  return cfg_constructs_;
}
std::list<Construct>& Function::constructs() { return cfg_constructs_; }

const BasicBlock* Function::first_block() const {
  if (ordered_blocks_.empty()) return nullptr;
  return ordered_blocks_[0];
}
BasicBlock* Function::first_block() {
  if (ordered_blocks_.empty()) return nullptr;
  return ordered_blocks_[0];
}

bool Function::IsBlockType(uint32_t merge_block_id, BlockType type) const {
  bool ret = false;
  const BasicBlock* block;
  std::tie(block, std::ignore) = GetBlock(merge_block_id);
  if (block) {
    ret = block->is_type(type);
  }
  return ret;
}

std::pair<const BasicBlock*, bool> Function::GetBlock(uint32_t block_id) const {
  const auto b = blocks_.find(block_id);
  if (b != end(blocks_)) {
    const BasicBlock* block = &(b->second);
    bool defined =
        undefined_blocks_.find(block->id()) == std::end(undefined_blocks_);
    return std::make_pair(block, defined);
  } else {
    return std::make_pair(nullptr, false);
  }
}

std::pair<BasicBlock*, bool> Function::GetBlock(uint32_t block_id) {
  const BasicBlock* out;
  bool defined;
  std::tie(out, defined) =
      const_cast<const Function*>(this)->GetBlock(block_id);
  return std::make_pair(const_cast<BasicBlock*>(out), defined);
}

Function::GetBlocksFunction Function::AugmentedCFGSuccessorsFunction() const {
  return [this](const BasicBlock* block) {
    auto where = augmented_successors_map_.find(block);
    return where == augmented_successors_map_.end() ? block->successors()
                                                    : &(*where).second;
  };
}

Function::GetBlocksFunction Function::AugmentedCFGPredecessorsFunction() const {
  return [this](const BasicBlock* block) {
    auto where = augmented_predecessors_map_.find(block);
    return where == augmented_predecessors_map_.end() ? block->predecessors()
                                                      : &(*where).second;
  };
}

Function::GetBlocksFunction Function::AugmentedStructuralCFGSuccessorsFunction()
    const {
  return [this](const BasicBlock* block) {
    auto where = augmented_successors_map_.find(block);
    return where == augmented_successors_map_.end()
               ? block->structural_successors()
               : &(*where).second;
  };
}

Function::GetBlocksFunction
Function::AugmentedStructuralCFGPredecessorsFunction() const {
  return [this](const BasicBlock* block) {
    auto where = augmented_predecessors_map_.find(block);
    return where == augmented_predecessors_map_.end()
               ? block->structural_predecessors()
               : &(*where).second;
  };
}

void Function::ComputeAugmentedCFG() {
  // Compute the successors of the pseudo-entry block, and
  // the predecessors of the pseudo exit block.
  auto succ_func = [](const BasicBlock* b) {
    return b->structural_successors();
  };
  auto pred_func = [](const BasicBlock* b) {
    return b->structural_predecessors();
  };
  CFA<BasicBlock>::ComputeAugmentedCFG(
      ordered_blocks_, &pseudo_entry_block_, &pseudo_exit_block_,
      &augmented_successors_map_, &augmented_predecessors_map_, succ_func,
      pred_func);
}

Construct& Function::AddConstruct(const Construct& new_construct) {
  cfg_constructs_.push_back(new_construct);
  auto& result = cfg_constructs_.back();
  entry_block_to_construct_[std::make_pair(new_construct.entry_block(),
                                           new_construct.type())] = &result;
  return result;
}

Construct& Function::FindConstructForEntryBlock(const BasicBlock* entry_block,
                                                ConstructType type) {
  auto where =
      entry_block_to_construct_.find(std::make_pair(entry_block, type));
  assert(where != entry_block_to_construct_.end());
  auto construct_ptr = (*where).second;
  assert(construct_ptr);
  return *construct_ptr;
}

int Function::GetBlockDepth(BasicBlock* bb) {
  // Guard against nullptr.
  if (!bb) {
    return 0;
  }
  // Only calculate the depth if it's not already calculated.
  // This function uses memoization to avoid duplicate CFG depth calculations.
  if (block_depth_.find(bb) != block_depth_.end()) {
    return block_depth_[bb];
  }
  // Avoid recursion. Something is wrong if the same block is encountered
  // multiple times.
  block_depth_[bb] = 0;

  BasicBlock* bb_dom = bb->immediate_dominator();
  if (!bb_dom || bb == bb_dom) {
    // This block has no dominator, so it's at depth 0.
    block_depth_[bb] = 0;
  } else if (bb->is_type(kBlockTypeContinue)) {
    // This rule must precede the rule for merge blocks in order to set up
    // depths correctly. If a block is both a merge and continue then the merge
    // is nested within the continue's loop (or the graph is incorrect).
    // The depth of the continue block entry point is 1 + loop header depth.
    Construct* continue_construct =
        entry_block_to_construct_[std::make_pair(bb, ConstructType::kContinue)];
    assert(continue_construct);
    // Continue construct has only 1 corresponding construct (loop header).
    Construct* loop_construct =
        continue_construct->corresponding_constructs()[0];
    assert(loop_construct);
    BasicBlock* loop_header = loop_construct->entry_block();
    // The continue target may be the loop itself (while 1).
    // In such cases, the depth of the continue block is: 1 + depth of the
    // loop's dominator block.
    if (loop_header == bb) {
      block_depth_[bb] = 1 + GetBlockDepth(bb_dom);
    } else {
      block_depth_[bb] = 1 + GetBlockDepth(loop_header);
    }
  } else if (bb->is_type(kBlockTypeMerge)) {
    // If this is a merge block, its depth is equal to the block before
    // branching.
    BasicBlock* header = merge_block_header_[bb];
    assert(header);
    block_depth_[bb] = GetBlockDepth(header);
  } else if (bb_dom->is_type(kBlockTypeSelection) ||
             bb_dom->is_type(kBlockTypeLoop)) {
    // The dominator of the given block is a header block. So, the nesting
    // depth of this block is: 1 + nesting depth of the header.
    block_depth_[bb] = 1 + GetBlockDepth(bb_dom);
  } else {
    block_depth_[bb] = GetBlockDepth(bb_dom);
  }
  return block_depth_[bb];
}

void Function::RegisterExecutionModelLimitation(spv::ExecutionModel model,
                                                const std::string& message) {
  execution_model_limitations_.push_back(
      [model, message](spv::ExecutionModel in_model, std::string* out_message) {
        if (model != in_model) {
          if (out_message) {
            *out_message = message;
          }
          return false;
        }
        return true;
      });
}

bool Function::IsCompatibleWithExecutionModel(spv::ExecutionModel model,
                                              std::string* reason) const {
  bool return_value = true;
  std::stringstream ss_reason;

  for (const auto& is_compatible : execution_model_limitations_) {
    std::string message;
    if (!is_compatible(model, &message)) {
      if (!reason) return false;
      return_value = false;
      if (!message.empty()) {
        ss_reason << message << "\n";
      }
    }
  }

  if (!return_value && reason) {
    *reason = ss_reason.str();
  }

  return return_value;
}

bool Function::CheckLimitations(const ValidationState_t& _,
                                const Function* entry_point,
                                std::string* reason) const {
  bool return_value = true;
  std::stringstream ss_reason;

  for (const auto& is_compatible : limitations_) {
    std::string message;
    if (!is_compatible(_, entry_point, &message)) {
      if (!reason) return false;
      return_value = false;
      if (!message.empty()) {
        ss_reason << message << "\n";
      }
    }
  }

  if (!return_value && reason) {
    *reason = ss_reason.str();
  }

  return return_value;
}

}  // namespace val
}  // namespace spvtools
