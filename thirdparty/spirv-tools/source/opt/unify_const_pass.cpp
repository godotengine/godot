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

#include "source/opt/unify_const_pass.h"

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "source/opt/def_use_manager.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace opt {
namespace {

// The trie that stores a bunch of result ids and, for a given instruction,
// searches the result id that has been defined with the same opcode, type and
// operands.
class ResultIdTrie {
 public:
  ResultIdTrie() : root_(new Node) {}

  // For a given instruction, extracts its opcode, type id and operand words
  // as an array of keys, looks up the trie to find a result id which is stored
  // with the same opcode, type id and operand words. If none of such result id
  // is found, creates a trie node with those keys, stores the instruction's
  // result id and returns that result id. If an existing result id is found,
  // returns the existing result id.
  uint32_t LookupEquivalentResultFor(const Instruction& inst) {
    auto keys = GetLookUpKeys(inst);
    auto* node = root_.get();
    for (uint32_t key : keys) {
      node = node->GetOrCreateTrieNodeFor(key);
    }
    if (node->result_id() == 0) {
      node->SetResultId(inst.result_id());
    }
    return node->result_id();
  }

 private:
  // The trie node to store result ids.
  class Node {
   public:
    using TrieNodeMap = std::unordered_map<uint32_t, std::unique_ptr<Node>>;

    Node() : result_id_(0), next_() {}
    uint32_t result_id() const { return result_id_; }

    // Sets the result id stored in this node.
    void SetResultId(uint32_t id) { result_id_ = id; }

    // Searches for the child trie node with the given key. If the node is
    // found, returns that node. Otherwise creates an empty child node with
    // that key and returns that newly created node.
    Node* GetOrCreateTrieNodeFor(uint32_t key) {
      auto iter = next_.find(key);
      if (iter == next_.end()) {
        // insert a new node and return the node.
        return next_.insert(std::make_pair(key, MakeUnique<Node>()))
            .first->second.get();
      }
      return iter->second.get();
    }

   private:
    // The result id stored in this node. 0 means this node is empty.
    uint32_t result_id_;
    // The mapping from the keys to the child nodes of this node.
    TrieNodeMap next_;
  };

  // Returns a vector of the opcode followed by the words in the raw SPIR-V
  // instruction encoding but without the result id.
  std::vector<uint32_t> GetLookUpKeys(const Instruction& inst) {
    std::vector<uint32_t> keys;
    // Need to use the opcode, otherwise there might be a conflict with the
    // following case when <op>'s binary value equals xx's id:
    //  OpSpecConstantOp tt <op> yy zz
    //  OpSpecConstantComposite tt xx yy zz;
    keys.push_back(static_cast<uint32_t>(inst.opcode()));
    for (const auto& operand : inst) {
      if (operand.type == SPV_OPERAND_TYPE_RESULT_ID) continue;
      keys.insert(keys.end(), operand.words.cbegin(), operand.words.cend());
    }
    return keys;
  }

  std::unique_ptr<Node> root_;  // The root node of the trie.
};
}  // namespace

Pass::Status UnifyConstantPass::Process() {
  bool modified = false;
  ResultIdTrie defined_constants;

  for (Instruction *next_instruction,
       *inst = &*(context()->types_values_begin());
       inst; inst = next_instruction) {
    next_instruction = inst->NextNode();

    // Do not handle the instruction when there are decorations upon the result
    // id.
    if (get_def_use_mgr()->GetAnnotations(inst->result_id()).size() != 0) {
      continue;
    }

    // The overall algorithm is to store the result ids of all the eligible
    // constants encountered so far in a trie. For a constant defining
    // instruction under consideration, use its opcode, result type id and
    // words in operands as an array of keys to lookup the trie. If a result id
    // can be found for that array of keys, a constant with exactly the same
    // value must has been defined before, the constant under processing
    // should be replaced by the constant previously defined. If no such result
    // id can be found for that array of keys, this must be the first time a
    // constant with its value be defined, we then create a new trie node to
    // store the result id with the keys. When replacing a duplicated constant
    // with a previously defined constant, all the uses of the duplicated
    // constant, which must be placed after the duplicated constant defining
    // instruction, will be updated. This way, the descendants of the
    // previously defined constant and the duplicated constant will both refer
    // to the previously defined constant. So that the operand ids which are
    // used in key arrays will be the ids of the unified constants, when
    // processing is up to a descendant. This makes comparing the key array
    // always valid for judging duplication.
    switch (inst->opcode()) {
      case spv::Op::OpConstantTrue:
      case spv::Op::OpConstantFalse:
      case spv::Op::OpConstant:
      case spv::Op::OpConstantNull:
      case spv::Op::OpConstantSampler:
      case spv::Op::OpConstantComposite:
      // Only spec constants defined with OpSpecConstantOp and
      // OpSpecConstantComposite should be processed in this pass. Spec
      // constants defined with OpSpecConstant{|True|False} are decorated with
      // 'SpecId' decoration and all of them should be treated as unique.
      // 'SpecId' is not applicable to SpecConstants defined with
      // OpSpecConstant{Op|Composite}, their values are not necessary to be
      // unique. When all the operands/components are the same between two
      // OpSpecConstant{Op|Composite} results, their result values must be the
      // same so are unifiable.
      case spv::Op::OpSpecConstantOp:
      case spv::Op::OpSpecConstantComposite: {
        uint32_t id = defined_constants.LookupEquivalentResultFor(*inst);
        if (id != inst->result_id()) {
          // The constant is a duplicated one, use the cached constant to
          // replace the uses of this duplicated one, then turn it to nop.
          context()->ReplaceAllUsesWith(inst->result_id(), id);
          context()->KillInst(inst);
          modified = true;
        }
        break;
      }
      default:
        break;
    }
  }
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
