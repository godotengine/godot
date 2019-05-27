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
 public:
  ConstantFoldingRules();

  // Returns true if there is at least 1 folding rule for |opcode|.
  bool HasFoldingRule(SpvOp opcode) const { return rules_.count(opcode); }

  // Returns an vector of constant folding rules for |opcode|.
  const std::vector<ConstantFoldingRule>& GetRulesForOpcode(
      SpvOp opcode) const {
    auto it = rules_.find(opcode);
    if (it != rules_.end()) {
      return it->second;
    }
    return empty_vector_;
  }

 private:
  std::unordered_map<uint32_t, std::vector<ConstantFoldingRule>> rules_;
  std::vector<ConstantFoldingRule> empty_vector_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_CONST_FOLDING_RULES_H_
