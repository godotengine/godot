// Copyright (c) 2021 Alastair F. Donaldson
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

#ifndef SOURCE_FUZZ_AVAILABLE_INSTRUCTIONS_H_
#define SOURCE_FUZZ_AVAILABLE_INSTRUCTIONS_H_

#include <unordered_map>
#include <vector>

#include "source/opt/instruction.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

// A class for allowing efficient querying of the instruction that satisfy a
// particular predicate that are available before a given instruction.
// Availability information is only computed for instructions in *reachable*
// basic blocks.
class AvailableInstructions {
 public:
  // The outer class captures availability information for a whole module, and
  // each instance of this inner class captures availability for a particular
  // instruction.
  class AvailableBeforeInstruction {
   public:
    AvailableBeforeInstruction(
        const AvailableInstructions& available_instructions,
        opt::Instruction* inst);

    // Returns the number of instructions that are available before the
    // instruction associated with this class.
    uint32_t size() const;

    // Returns true if and only if |size()| is 0.
    bool empty() const;

    // Requires |index| < |size()|. Returns the ith available instruction.
    opt::Instruction* operator[](uint32_t index) const;

   private:
    // A references to an instance of the outer class.
    const AvailableInstructions& available_instructions_;

    // The instruction for which availability information is captured.
    opt::Instruction* inst_;

    // A cache to improve the efficiency of the [] operator. The [] operator
    // requires walking the instruction's dominator tree to find an instruction
    // at a particular index, which is a linear time operation. By inserting all
    // instructions that are traversed during this search into a cache, future
    // lookups will take constant time unless they require traversing the
    // dominator tree more deeply.
    mutable std::unordered_map<uint32_t, opt::Instruction*> index_cache;
  };

  // Constructs availability instructions for |ir_context|, where instructions
  // are only available if they satisfy |predicate|.
  AvailableInstructions(
      opt::IRContext* ir_context,
      const std::function<bool(opt::IRContext*, opt::Instruction*)>& predicate);

  // Yields instruction availability for |inst|.
  AvailableBeforeInstruction GetAvailableBeforeInstruction(
      opt::Instruction* inst) const;

 private:
  // The module in which all instructions are contained.
  opt::IRContext* ir_context_;

  // The global instructions that satisfy the predicate.
  std::vector<opt::Instruction*> available_globals_;

  // Per function, the parameters that satisfy the predicate.
  std::unordered_map<opt::Function*, std::vector<opt::Instruction*>>
      available_params_;

  // The number of instructions that satisfy the predicate and that are
  // available at the entry to a block. For the entry block of a function this
  // is the number of available globals + the number of available function
  // parameters. For any other block it is the number of available instructions
  // for the blocks immediate dominator + the number of instructions generated
  // by the immediate dominator.
  std::unordered_map<opt::BasicBlock*, uint32_t> num_available_at_block_entry_;

  // For each block this records those instructions in the block that satisfy
  // the predicate.
  std::unordered_map<opt::BasicBlock*, std::vector<opt::Instruction*>>
      generated_by_block_;

  // For each instruction this records how many instructions satisfying the
  // predicate are available before the instruction.
  std::unordered_map<opt::Instruction*, uint32_t>
      num_available_before_instruction_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_AVAILABLE_INSTRUCTIONS_H_
