// Copyright (c) 2018 Google LLC
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

#ifndef SOURCE_OPT_CONST_FOLDING_RULES_H_
#define SOURCE_OPT_CONST_FOLDING_RULES_H_

#include <unordered_map>
#include <vector>

#include "source/opt/constants.h"

namespace spvtools {
namespace opt {

// Constant Folding Rules:
//
// The folding mechanism is built around the concept of a |ConstantFoldingRule|.
// A constant folding rule is a function that implements a method of simplifying
// an instruction to a constant.
//
// The inputs to a folding rule are:
//     |inst| - the instruction to be simplified.
//     |constants| - if an in-operands is an id of a constant, then the
//                   corresponding value in |constants| contains that
//                   constant value.  Otherwise, the corresponding entry in
//                   |constants| is |nullptr|.
//
// A constant folding rule returns a pointer to an Constant if |inst| can be
// simplified using this rule. Otherwise, it returns |nullptr|.
//
// See const_folding_rules.cpp for examples on how to write a constant folding
// rule.
//
// Be sure to add new constant folding rules to the table of constant folding
// rules in the constructor for ConstantFoldingRules.  The new rule should be
// added to the list for every opcode that it applies to.  Note that earlier
// rules in the list are given priority.  That is, if an earlier rule is able to
// fold an instruction, the later rules will not be attempted.

using ConstantFoldingRule = std::function<const analysis::Constant*(
    IRContext* ctx, Instruction* inst,
    const std::vector<const analysis::Constant*>& constants)>;

class ConstantFoldingRules {
 protected:
  // The |Key| and |Value| structs are used to by-pass a "decorated name length
  // exceeded, name was truncated" warning on VS2013 and VS2015.
  struct Key {
    uint32_t instruction_set;
    uint32_t opcode;
  };

  friend bool operator<(const Key& a, const Key& b) {
    if (a.instruction_set < b.instruction_set) {
      return true;
    }
    if (a.instruction_set > b.instruction_set) {
      return false;
    }
    return a.opcode < b.opcode;
  }

  struct Value {
    std::vector<ConstantFoldingRule> value;
    void push_back(ConstantFoldingRule rule) { value.push_back(rule); }
  };

 public:
  ConstantFoldingRules(IRContext* ctx) : context_(ctx) {}
  virtual ~ConstantFoldingRules() = default;

  // Returns true if there is at least 1 folding rule for |opcode|.
  bool HasFoldingRule(const Instruction* inst) const {
    return !GetRulesForInstruction(inst).empty();
  }

  // Returns true if there is at least 1 folding rule for |inst|.
  const std::vector<ConstantFoldingRule>& GetRulesForInstruction(
      const Instruction* inst) const {
    if (inst->opcode() != spv::Op::OpExtInst) {
      auto it = rules_.find(inst->opcode());
      if (it != rules_.end()) {
        return it->second.value;
      }
    } else {
      uint32_t ext_inst_id = inst->GetSingleWordInOperand(0);
      uint32_t ext_opcode = inst->GetSingleWordInOperand(1);
      auto it = ext_rules_.find({ext_inst_id, ext_opcode});
      if (it != ext_rules_.end()) {
        return it->second.value;
      }
    }
    return empty_vector_;
  }

  // Add the folding rules.
  virtual void AddFoldingRules();

 protected:
  struct hasher {
    size_t operator()(const spv::Op& op) const noexcept {
      return std::hash<uint32_t>()(uint32_t(op));
    }
  };

  // |rules[opcode]| is the set of rules that can be applied to instructions
  // with |opcode| as the opcode.
  std::unordered_map<spv::Op, Value, hasher> rules_;

  // The folding rules for extended instructions.
  std::map<Key, Value> ext_rules_;

 private:
  // The context that the instruction to be folded will be a part of.
  IRContext* context_;

  // The empty set of rules to be used as the default return value in
  // |GetRulesForInstruction|.
  std::vector<ConstantFoldingRule> empty_vector_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_CONST_FOLDING_RULES_H_
