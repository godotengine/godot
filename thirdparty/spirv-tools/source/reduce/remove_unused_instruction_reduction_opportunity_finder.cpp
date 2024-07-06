// Copyright (c) 2018 Google Inc.
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

#include "source/reduce/remove_unused_instruction_reduction_opportunity_finder.h"

#include "source/opcode.h"
#include "source/opt/instruction.h"
#include "source/reduce/remove_instruction_reduction_opportunity.h"

namespace spvtools {
namespace reduce {

RemoveUnusedInstructionReductionOpportunityFinder::
    RemoveUnusedInstructionReductionOpportunityFinder(
        bool remove_constants_and_undefs)
    : remove_constants_and_undefs_(remove_constants_and_undefs) {}

std::vector<std::unique_ptr<ReductionOpportunity>>
RemoveUnusedInstructionReductionOpportunityFinder::GetAvailableOpportunities(
    opt::IRContext* context, uint32_t target_function) const {
  std::vector<std::unique_ptr<ReductionOpportunity>> result;

  if (!target_function) {
    // We are not restricting reduction to a specific function, so we consider
    // unused instructions defined outside functions.

    for (auto& inst : context->module()->debugs1()) {
      if (context->get_def_use_mgr()->NumUses(&inst) > 0) {
        continue;
      }
      result.push_back(
          MakeUnique<RemoveInstructionReductionOpportunity>(&inst));
    }

    for (auto& inst : context->module()->debugs2()) {
      if (context->get_def_use_mgr()->NumUses(&inst) > 0) {
        continue;
      }
      result.push_back(
          MakeUnique<RemoveInstructionReductionOpportunity>(&inst));
    }

    for (auto& inst : context->module()->debugs3()) {
      if (context->get_def_use_mgr()->NumUses(&inst) > 0) {
        continue;
      }
      result.push_back(
          MakeUnique<RemoveInstructionReductionOpportunity>(&inst));
    }

    for (auto& inst : context->module()->ext_inst_debuginfo()) {
      if (context->get_def_use_mgr()->NumUses(&inst) > 0) {
        continue;
      }
      result.push_back(
          MakeUnique<RemoveInstructionReductionOpportunity>(&inst));
    }

    for (auto& inst : context->module()->types_values()) {
      if (!remove_constants_and_undefs_ &&
          spvOpcodeIsConstantOrUndef(inst.opcode())) {
        continue;
      }
      if (!OnlyReferencedByIntimateDecorationOrEntryPointInterface(context,
                                                                   inst)) {
        continue;
      }
      result.push_back(
          MakeUnique<RemoveInstructionReductionOpportunity>(&inst));
    }

    for (auto& inst : context->module()->annotations()) {
      if (context->get_def_use_mgr()->NumUsers(&inst) > 0) {
        continue;
      }
      if (!IsIndependentlyRemovableDecoration(inst)) {
        continue;
      }
      result.push_back(
          MakeUnique<RemoveInstructionReductionOpportunity>(&inst));
    }
  }

  for (auto* function : GetTargetFunctions(context, target_function)) {
    for (auto& block : *function) {
      for (auto& inst : block) {
        if (context->get_def_use_mgr()->NumUses(&inst) > 0) {
          continue;
        }
        if (!remove_constants_and_undefs_ &&
            spvOpcodeIsConstantOrUndef(inst.opcode())) {
          continue;
        }
        if (spvOpcodeIsBlockTerminator(inst.opcode()) ||
            inst.opcode() == spv::Op::OpSelectionMerge ||
            inst.opcode() == spv::Op::OpLoopMerge) {
          // In this reduction pass we do not want to affect static
          // control flow.
          continue;
        }
        // Given that we're in a block, we should only get here if
        // the instruction is not directly related to control flow;
        // i.e., it's some straightforward instruction with an
        // unused result, like an arithmetic operation or function
        // call.
        result.push_back(
            MakeUnique<RemoveInstructionReductionOpportunity>(&inst));
      }
    }
  }
  return result;
}

std::string RemoveUnusedInstructionReductionOpportunityFinder::GetName() const {
  return "RemoveUnusedInstructionReductionOpportunityFinder";
}

bool RemoveUnusedInstructionReductionOpportunityFinder::
    OnlyReferencedByIntimateDecorationOrEntryPointInterface(
        opt::IRContext* context, const opt::Instruction& inst) const {
  return context->get_def_use_mgr()->WhileEachUse(
      &inst, [this](opt::Instruction* user, uint32_t use_index) -> bool {
        return (user->IsDecoration() &&
                !IsIndependentlyRemovableDecoration(*user)) ||
               (user->opcode() == spv::Op::OpEntryPoint && use_index > 2);
      });
}

bool RemoveUnusedInstructionReductionOpportunityFinder::
    IsIndependentlyRemovableDecoration(const opt::Instruction& inst) const {
  uint32_t decoration;
  switch (inst.opcode()) {
    case spv::Op::OpDecorate:
    case spv::Op::OpDecorateId:
    case spv::Op::OpDecorateString:
      decoration = inst.GetSingleWordInOperand(1u);
      break;
    case spv::Op::OpMemberDecorate:
    case spv::Op::OpMemberDecorateString:
      decoration = inst.GetSingleWordInOperand(2u);
      break;
    default:
      // The instruction is not a decoration.  It is legitimate for this to be
      // reached: it allows the method to be invoked on arbitrary instructions.
      return false;
  }

  // We conservatively only remove specific decorations that we believe will
  // not change the shader interface, will not make the shader invalid, will
  // actually be found in practice, etc.

  switch (spv::Decoration(decoration)) {
    case spv::Decoration::RelaxedPrecision:
    case spv::Decoration::NoSignedWrap:
    case spv::Decoration::NoContraction:
    case spv::Decoration::NoUnsignedWrap:
    case spv::Decoration::UserSemantic:
      return true;
    default:
      return false;
  }
}

}  // namespace reduce
}  // namespace spvtools
