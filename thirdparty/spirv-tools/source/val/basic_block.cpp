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

#include "source/val/basic_block.h"

#include <algorithm>
#include <vector>

namespace spvtools {
namespace val {

BasicBlock::BasicBlock(uint32_t label_id)
    : id_(label_id),
      immediate_dominator_(nullptr),
      immediate_structural_dominator_(nullptr),
      immediate_structural_post_dominator_(nullptr),
      predecessors_(),
      successors_(),
      type_(0),
      reachable_(false),
      structurally_reachable_(false),
      label_(nullptr),
      terminator_(nullptr) {}

void BasicBlock::SetImmediateDominator(BasicBlock* dom_block) {
  immediate_dominator_ = dom_block;
}

void BasicBlock::SetImmediateStructuralDominator(BasicBlock* dom_block) {
  immediate_structural_dominator_ = dom_block;
}

void BasicBlock::SetImmediateStructuralPostDominator(BasicBlock* pdom_block) {
  immediate_structural_post_dominator_ = pdom_block;
}

const BasicBlock* BasicBlock::immediate_dominator() const {
  return immediate_dominator_;
}

const BasicBlock* BasicBlock::immediate_structural_dominator() const {
  return immediate_structural_dominator_;
}

const BasicBlock* BasicBlock::immediate_structural_post_dominator() const {
  return immediate_structural_post_dominator_;
}

BasicBlock* BasicBlock::immediate_dominator() { return immediate_dominator_; }
BasicBlock* BasicBlock::immediate_structural_dominator() {
  return immediate_structural_dominator_;
}
BasicBlock* BasicBlock::immediate_structural_post_dominator() {
  return immediate_structural_post_dominator_;
}

void BasicBlock::RegisterSuccessors(
    const std::vector<BasicBlock*>& next_blocks) {
  for (auto& block : next_blocks) {
    block->predecessors_.push_back(this);
    successors_.push_back(block);

    // Register structural successors/predecessors too.
    block->structural_predecessors_.push_back(this);
    structural_successors_.push_back(block);
  }
}

bool BasicBlock::dominates(const BasicBlock& other) const {
  return (this == &other) ||
         !(other.dom_end() ==
           std::find(other.dom_begin(), other.dom_end(), this));
}

bool BasicBlock::structurally_dominates(const BasicBlock& other) const {
  return (this == &other) || !(other.structural_dom_end() ==
                               std::find(other.structural_dom_begin(),
                                         other.structural_dom_end(), this));
}

bool BasicBlock::structurally_postdominates(const BasicBlock& other) const {
  return (this == &other) || !(other.structural_pdom_end() ==
                               std::find(other.structural_pdom_begin(),
                                         other.structural_pdom_end(), this));
}

BasicBlock::DominatorIterator::DominatorIterator() : current_(nullptr) {}

BasicBlock::DominatorIterator::DominatorIterator(
    const BasicBlock* block,
    std::function<const BasicBlock*(const BasicBlock*)> dominator_func)
    : current_(block), dom_func_(dominator_func) {}

BasicBlock::DominatorIterator& BasicBlock::DominatorIterator::operator++() {
  if (current_ == dom_func_(current_)) {
    current_ = nullptr;
  } else {
    current_ = dom_func_(current_);
  }
  return *this;
}

const BasicBlock::DominatorIterator BasicBlock::dom_begin() const {
  return DominatorIterator(
      this, [](const BasicBlock* b) { return b->immediate_dominator(); });
}

BasicBlock::DominatorIterator BasicBlock::dom_begin() {
  return DominatorIterator(
      this, [](const BasicBlock* b) { return b->immediate_dominator(); });
}

const BasicBlock::DominatorIterator BasicBlock::dom_end() const {
  return DominatorIterator();
}

BasicBlock::DominatorIterator BasicBlock::dom_end() {
  return DominatorIterator();
}

const BasicBlock::DominatorIterator BasicBlock::structural_dom_begin() const {
  return DominatorIterator(this, [](const BasicBlock* b) {
    return b->immediate_structural_dominator();
  });
}

BasicBlock::DominatorIterator BasicBlock::structural_dom_begin() {
  return DominatorIterator(this, [](const BasicBlock* b) {
    return b->immediate_structural_dominator();
  });
}

const BasicBlock::DominatorIterator BasicBlock::structural_dom_end() const {
  return DominatorIterator();
}

BasicBlock::DominatorIterator BasicBlock::structural_dom_end() {
  return DominatorIterator();
}

const BasicBlock::DominatorIterator BasicBlock::structural_pdom_begin() const {
  return DominatorIterator(this, [](const BasicBlock* b) {
    return b->immediate_structural_post_dominator();
  });
}

BasicBlock::DominatorIterator BasicBlock::structural_pdom_begin() {
  return DominatorIterator(this, [](const BasicBlock* b) {
    return b->immediate_structural_post_dominator();
  });
}

const BasicBlock::DominatorIterator BasicBlock::structural_pdom_end() const {
  return DominatorIterator();
}

BasicBlock::DominatorIterator BasicBlock::structural_pdom_end() {
  return DominatorIterator();
}

bool operator==(const BasicBlock::DominatorIterator& lhs,
                const BasicBlock::DominatorIterator& rhs) {
  return lhs.current_ == rhs.current_;
}

bool operator!=(const BasicBlock::DominatorIterator& lhs,
                const BasicBlock::DominatorIterator& rhs) {
  return !(lhs == rhs);
}

const BasicBlock*& BasicBlock::DominatorIterator::operator*() {
  return current_;
}

}  // namespace val
}  // namespace spvtools
