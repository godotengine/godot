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

#include <cassert>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "source/cfa.h"
#include "source/opcode.h"
#include "source/spirv_constant.h"
#include "source/spirv_validator_options.h"
#include "source/val/basic_block.h"
#include "source/val/construct.h"
#include "source/val/function.h"
#include "source/val/validate.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

spv_result_t ValidatePhi(ValidationState_t& _, const Instruction* inst) {
  auto block = inst->block();
  size_t num_in_ops = inst->words().size() - 3;
  if (num_in_ops % 2 != 0) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpPhi does not have an equal number of incoming values and "
              "basic blocks.";
  }

  if (_.IsVoidType(inst->type_id())) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "OpPhi must not have void result type";
  }
  if (_.IsPointerType(inst->type_id()) &&
      _.addressing_model() == spv::AddressingModel::Logical) {
    if (!_.features().variable_pointers) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Using pointers with OpPhi requires capability "
             << "VariablePointers or VariablePointersStorageBuffer";
    }
  }

  const Instruction* type_inst = _.FindDef(inst->type_id());
  assert(type_inst);
  const spv::Op type_opcode = type_inst->opcode();

  if (!_.options()->before_hlsl_legalization &&
      !_.HasCapability(spv::Capability::BindlessTextureNV)) {
    if (type_opcode == spv::Op::OpTypeSampledImage ||
        (_.HasCapability(spv::Capability::Shader) &&
         (type_opcode == spv::Op::OpTypeImage ||
          type_opcode == spv::Op::OpTypeSampler))) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "Result type cannot be Op" << spvOpcodeString(type_opcode);
    }
  }

  // Create a uniqued vector of predecessor ids for comparison against
  // incoming values. OpBranchConditional %cond %label %label produces two
  // predecessors in the CFG.
  std::vector<uint32_t> pred_ids;
  std::transform(block->predecessors()->begin(), block->predecessors()->end(),
                 std::back_inserter(pred_ids),
                 [](const BasicBlock* b) { return b->id(); });
  std::sort(pred_ids.begin(), pred_ids.end());
  pred_ids.erase(std::unique(pred_ids.begin(), pred_ids.end()), pred_ids.end());

  size_t num_edges = num_in_ops / 2;
  if (num_edges != pred_ids.size()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpPhi's number of incoming blocks (" << num_edges
           << ") does not match block's predecessor count ("
           << block->predecessors()->size() << ").";
  }

  std::unordered_set<uint32_t> observed_predecessors;

  for (size_t i = 3; i < inst->words().size(); ++i) {
    auto inc_id = inst->word(i);
    if (i % 2 == 1) {
      // Incoming value type must match the phi result type.
      auto inc_type_id = _.GetTypeId(inc_id);
      if (inst->type_id() != inc_type_id) {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << "OpPhi's result type <id> " << _.getIdName(inst->type_id())
               << " does not match incoming value <id> " << _.getIdName(inc_id)
               << " type <id> " << _.getIdName(inc_type_id) << ".";
      }
    } else {
      if (_.GetIdOpcode(inc_id) != spv::Op::OpLabel) {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << "OpPhi's incoming basic block <id> " << _.getIdName(inc_id)
               << " is not an OpLabel.";
      }

      // Incoming basic block must be an immediate predecessor of the phi's
      // block.
      if (!std::binary_search(pred_ids.begin(), pred_ids.end(), inc_id)) {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << "OpPhi's incoming basic block <id> " << _.getIdName(inc_id)
               << " is not a predecessor of <id> " << _.getIdName(block->id())
               << ".";
      }

      // We must not have already seen this predecessor as one of the phi's
      // operands.
      if (observed_predecessors.count(inc_id) != 0) {
        return _.diag(SPV_ERROR_INVALID_ID, inst)
               << "OpPhi references incoming basic block <id> "
               << _.getIdName(inc_id) << " multiple times.";
      }

      // Note the fact that we have now observed this predecessor.
      observed_predecessors.insert(inc_id);
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateBranch(ValidationState_t& _, const Instruction* inst) {
  // target operands must be OpLabel
  const auto id = inst->GetOperandAs<uint32_t>(0);
  const auto target = _.FindDef(id);
  if (!target || spv::Op::OpLabel != target->opcode()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "'Target Label' operands for OpBranch must be the ID "
              "of an OpLabel instruction";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateBranchConditional(ValidationState_t& _,
                                       const Instruction* inst) {
  // num_operands is either 3 or 5 --- if 5, the last two need to be literal
  // integers
  const auto num_operands = inst->operands().size();
  if (num_operands != 3 && num_operands != 5) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpBranchConditional requires either 3 or 5 parameters";
  }

  // grab the condition operand and check that it is a bool
  const auto cond_id = inst->GetOperandAs<uint32_t>(0);
  const auto cond_op = _.FindDef(cond_id);
  if (!cond_op || !cond_op->type_id() ||
      !_.IsBoolScalarType(cond_op->type_id())) {
    return _.diag(SPV_ERROR_INVALID_ID, inst) << "Condition operand for "
                                                 "OpBranchConditional must be "
                                                 "of boolean type";
  }

  // target operands must be OpLabel
  // note that we don't need to check that the target labels are in the same
  // function,
  // PerformCfgChecks already checks for that
  const auto true_id = inst->GetOperandAs<uint32_t>(1);
  const auto true_target = _.FindDef(true_id);
  if (!true_target || spv::Op::OpLabel != true_target->opcode()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "The 'True Label' operand for OpBranchConditional must be the "
              "ID of an OpLabel instruction";
  }

  const auto false_id = inst->GetOperandAs<uint32_t>(2);
  const auto false_target = _.FindDef(false_id);
  if (!false_target || spv::Op::OpLabel != false_target->opcode()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "The 'False Label' operand for OpBranchConditional must be the "
              "ID of an OpLabel instruction";
  }

  // A similar requirement for SPV_KHR_maximal_reconvergence is deferred until
  // entry point call trees have been reconrded.
  if (_.version() >= SPV_SPIRV_VERSION_WORD(1, 6) && true_id == false_id) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "In SPIR-V 1.6 or later, True Label and False Label must be "
              "different labels";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateSwitch(ValidationState_t& _, const Instruction* inst) {
  const auto num_operands = inst->operands().size();
  // At least two operands (selector, default), any more than that are
  // literal/target.

  const auto sel_type_id = _.GetOperandTypeId(inst, 0);
  if (!_.IsIntScalarType(sel_type_id)) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Selector type must be OpTypeInt";
  }

  const auto default_label = _.FindDef(inst->GetOperandAs<uint32_t>(1));
  if (default_label->opcode() != spv::Op::OpLabel) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Default must be an OpLabel instruction";
  }

  // target operands must be OpLabel
  for (size_t i = 2; i < num_operands; i += 2) {
    // literal, id
    const auto id = inst->GetOperandAs<uint32_t>(i + 1);
    const auto target = _.FindDef(id);
    if (!target || spv::Op::OpLabel != target->opcode()) {
      return _.diag(SPV_ERROR_INVALID_ID, inst)
             << "'Target Label' operands for OpSwitch must be IDs of an "
                "OpLabel instruction";
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateReturnValue(ValidationState_t& _,
                                 const Instruction* inst) {
  const auto value_id = inst->GetOperandAs<uint32_t>(0);
  const auto value = _.FindDef(value_id);
  if (!value || !value->type_id()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpReturnValue Value <id> " << _.getIdName(value_id)
           << " does not represent a value.";
  }
  auto value_type = _.FindDef(value->type_id());
  if (!value_type || spv::Op::OpTypeVoid == value_type->opcode()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpReturnValue value's type <id> "
           << _.getIdName(value->type_id()) << " is missing or void.";
  }

  if (_.addressing_model() == spv::AddressingModel::Logical &&
      spv::Op::OpTypePointer == value_type->opcode() &&
      !_.features().variable_pointers && !_.options()->relax_logical_pointer) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpReturnValue value's type <id> "
           << _.getIdName(value->type_id())
           << " is a pointer, which is invalid in the Logical addressing "
              "model.";
  }

  const auto function = inst->function();
  const auto return_type = _.FindDef(function->GetResultTypeId());
  if (!return_type || return_type->id() != value_type->id()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpReturnValue Value <id> " << _.getIdName(value_id)
           << "s type does not match OpFunction's return type.";
  }

  return SPV_SUCCESS;
}

uint32_t operator>>(const spv::LoopControlShift& lhs,
                    const spv::LoopControlShift& rhs) {
  return uint32_t(lhs) >> uint32_t(rhs);
}

spv_result_t ValidateLoopMerge(ValidationState_t& _, const Instruction* inst) {
  const auto merge_id = inst->GetOperandAs<uint32_t>(0);
  const auto merge = _.FindDef(merge_id);
  if (!merge || merge->opcode() != spv::Op::OpLabel) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Merge Block " << _.getIdName(merge_id) << " must be an OpLabel";
  }
  if (merge_id == inst->block()->id()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Merge Block may not be the block containing the OpLoopMerge\n";
  }

  const auto continue_id = inst->GetOperandAs<uint32_t>(1);
  const auto continue_target = _.FindDef(continue_id);
  if (!continue_target || continue_target->opcode() != spv::Op::OpLabel) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Continue Target " << _.getIdName(continue_id)
           << " must be an OpLabel";
  }

  if (merge_id == continue_id) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "Merge Block and Continue Target must be different ids";
  }

  const auto loop_control = inst->GetOperandAs<spv::LoopControlShift>(2);
  if ((loop_control >> spv::LoopControlShift::Unroll) & 0x1 &&
      (loop_control >> spv::LoopControlShift::DontUnroll) & 0x1) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst)
           << "Unroll and DontUnroll loop controls must not both be specified";
  }
  if ((loop_control >> spv::LoopControlShift::DontUnroll) & 0x1 &&
      (loop_control >> spv::LoopControlShift::PeelCount) & 0x1) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst) << "PeelCount and DontUnroll "
                                                   "loop controls must not "
                                                   "both be specified";
  }
  if ((loop_control >> spv::LoopControlShift::DontUnroll) & 0x1 &&
      (loop_control >> spv::LoopControlShift::PartialCount) & 0x1) {
    return _.diag(SPV_ERROR_INVALID_DATA, inst) << "PartialCount and "
                                                   "DontUnroll loop controls "
                                                   "must not both be specified";
  }

  uint32_t operand = 3;
  if ((loop_control >> spv::LoopControlShift::DependencyLength) & 0x1) {
    ++operand;
  }
  if ((loop_control >> spv::LoopControlShift::MinIterations) & 0x1) {
    ++operand;
  }
  if ((loop_control >> spv::LoopControlShift::MaxIterations) & 0x1) {
    ++operand;
  }
  if ((loop_control >> spv::LoopControlShift::IterationMultiple) & 0x1) {
    if (inst->operands().size() < operand ||
        inst->GetOperandAs<uint32_t>(operand) == 0) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst) << "IterationMultiple loop "
                                                     "control operand must be "
                                                     "greater than zero";
    }
    ++operand;
  }
  if ((loop_control >> spv::LoopControlShift::PeelCount) & 0x1) {
    ++operand;
  }
  if ((loop_control >> spv::LoopControlShift::PartialCount) & 0x1) {
    ++operand;
  }

  // That the right number of operands is present is checked by the parser. The
  // above code tracks operands for expanded validation checking in the future.

  return SPV_SUCCESS;
}

}  // namespace

void printDominatorList(const BasicBlock& b) {
  std::cout << b.id() << " is dominated by: ";
  const BasicBlock* bb = &b;
  while (bb->immediate_dominator() != bb) {
    bb = bb->immediate_dominator();
    std::cout << bb->id() << " ";
  }
}

#define CFG_ASSERT(ASSERT_FUNC, TARGET) \
  if (spv_result_t rcode = ASSERT_FUNC(_, TARGET)) return rcode

spv_result_t FirstBlockAssert(ValidationState_t& _, uint32_t target) {
  if (_.current_function().IsFirstBlock(target)) {
    return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef(_.current_function().id()))
           << "First block " << _.getIdName(target) << " of function "
           << _.getIdName(_.current_function().id()) << " is targeted by block "
           << _.getIdName(_.current_function().current_block()->id());
  }
  return SPV_SUCCESS;
}

spv_result_t MergeBlockAssert(ValidationState_t& _, uint32_t merge_block) {
  if (_.current_function().IsBlockType(merge_block, kBlockTypeMerge)) {
    return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef(_.current_function().id()))
           << "Block " << _.getIdName(merge_block)
           << " is already a merge block for another header";
  }
  return SPV_SUCCESS;
}

/// Update the continue construct's exit blocks once the backedge blocks are
/// identified in the CFG.
void UpdateContinueConstructExitBlocks(
    Function& function,
    const std::vector<std::pair<uint32_t, uint32_t>>& back_edges) {
  auto& constructs = function.constructs();
  // TODO(umar): Think of a faster way to do this
  for (auto& edge : back_edges) {
    uint32_t back_edge_block_id;
    uint32_t loop_header_block_id;
    std::tie(back_edge_block_id, loop_header_block_id) = edge;
    auto is_this_header = [=](Construct& c) {
      return c.type() == ConstructType::kLoop &&
             c.entry_block()->id() == loop_header_block_id;
    };

    for (auto construct : constructs) {
      if (is_this_header(construct)) {
        Construct* continue_construct =
            construct.corresponding_constructs().back();
        assert(continue_construct->type() == ConstructType::kContinue);

        BasicBlock* back_edge_block;
        std::tie(back_edge_block, std::ignore) =
            function.GetBlock(back_edge_block_id);
        continue_construct->set_exit(back_edge_block);
      }
    }
  }
}

std::tuple<std::string, std::string, std::string> ConstructNames(
    ConstructType type) {
  std::string construct_name, header_name, exit_name;

  switch (type) {
    case ConstructType::kSelection:
      construct_name = "selection";
      header_name = "selection header";
      exit_name = "merge block";
      break;
    case ConstructType::kLoop:
      construct_name = "loop";
      header_name = "loop header";
      exit_name = "merge block";
      break;
    case ConstructType::kContinue:
      construct_name = "continue";
      header_name = "continue target";
      exit_name = "back-edge block";
      break;
    case ConstructType::kCase:
      construct_name = "case";
      header_name = "case entry block";
      exit_name = "case exit block";
      break;
    default:
      assert(1 == 0 && "Not defined type");
  }

  return std::make_tuple(construct_name, header_name, exit_name);
}

/// Constructs an error message for construct validation errors
std::string ConstructErrorString(const Construct& construct,
                                 const std::string& header_string,
                                 const std::string& exit_string,
                                 const std::string& dominate_text) {
  std::string construct_name, header_name, exit_name;
  std::tie(construct_name, header_name, exit_name) =
      ConstructNames(construct.type());

  // TODO(umar): Add header block for continue constructs to error message
  return "The " + construct_name + " construct with the " + header_name + " " +
         header_string + " " + dominate_text + " the " + exit_name + " " +
         exit_string;
}

// Finds the fall through case construct of |target_block| and records it in
// |case_fall_through|. Returns SPV_ERROR_INVALID_CFG if the case construct
// headed by |target_block| branches to multiple case constructs.
spv_result_t FindCaseFallThrough(
    ValidationState_t& _, BasicBlock* target_block, uint32_t* case_fall_through,
    const BasicBlock* merge, const std::unordered_set<uint32_t>& case_targets,
    Function* function) {
  std::vector<BasicBlock*> stack;
  stack.push_back(target_block);
  std::unordered_set<const BasicBlock*> visited;
  bool target_reachable = target_block->structurally_reachable();
  int target_depth = function->GetBlockDepth(target_block);
  while (!stack.empty()) {
    auto block = stack.back();
    stack.pop_back();

    if (block == merge) continue;

    if (!visited.insert(block).second) continue;

    if (target_reachable && block->structurally_reachable() &&
        target_block->structurally_dominates(*block)) {
      // Still in the case construct.
      for (auto successor : *block->successors()) {
        stack.push_back(successor);
      }
    } else {
      // Exiting the case construct to non-merge block.
      if (!case_targets.count(block->id())) {
        int depth = function->GetBlockDepth(block);
        if ((depth < target_depth) ||
            (depth == target_depth && block->is_type(kBlockTypeContinue))) {
          continue;
        }

        return _.diag(SPV_ERROR_INVALID_CFG, target_block->label())
               << "Case construct that targets "
               << _.getIdName(target_block->id())
               << " has invalid branch to block " << _.getIdName(block->id())
               << " (not another case construct, corresponding merge, outer "
                  "loop merge or outer loop continue)";
      }

      if (*case_fall_through == 0u) {
        if (target_block != block) {
          *case_fall_through = block->id();
        }
      } else if (*case_fall_through != block->id()) {
        // Case construct has at most one branch to another case construct.
        return _.diag(SPV_ERROR_INVALID_CFG, target_block->label())
               << "Case construct that targets "
               << _.getIdName(target_block->id())
               << " has branches to multiple other case construct targets "
               << _.getIdName(*case_fall_through) << " and "
               << _.getIdName(block->id());
      }
    }
  }

  return SPV_SUCCESS;
}

spv_result_t StructuredSwitchChecks(ValidationState_t& _, Function* function,
                                    const Instruction* switch_inst,
                                    const BasicBlock* header,
                                    const BasicBlock* merge) {
  std::unordered_set<uint32_t> case_targets;
  for (uint32_t i = 1; i < switch_inst->operands().size(); i += 2) {
    uint32_t target = switch_inst->GetOperandAs<uint32_t>(i);
    if (target != merge->id()) case_targets.insert(target);
  }
  // Tracks how many times each case construct is targeted by another case
  // construct.
  std::map<uint32_t, uint32_t> num_fall_through_targeted;
  uint32_t default_case_fall_through = 0u;
  uint32_t default_target = switch_inst->GetOperandAs<uint32_t>(1u);
  bool default_appears_multiple_times = false;
  for (uint32_t i = 3; i < switch_inst->operands().size(); i += 2) {
    if (default_target == switch_inst->GetOperandAs<uint32_t>(i)) {
      default_appears_multiple_times = true;
      break;
    }
  }
  std::unordered_map<uint32_t, uint32_t> seen_to_fall_through;
  for (uint32_t i = 1; i < switch_inst->operands().size(); i += 2) {
    uint32_t target = switch_inst->GetOperandAs<uint32_t>(i);
    if (target == merge->id()) continue;

    uint32_t case_fall_through = 0u;
    auto seen_iter = seen_to_fall_through.find(target);
    if (seen_iter == seen_to_fall_through.end()) {
      const auto target_block = function->GetBlock(target).first;
      // OpSwitch must dominate all its case constructs.
      if (header->structurally_reachable() &&
          target_block->structurally_reachable() &&
          !header->structurally_dominates(*target_block)) {
        return _.diag(SPV_ERROR_INVALID_CFG, header->label())
               << "Switch header " << _.getIdName(header->id())
               << " does not structurally dominate its case construct "
               << _.getIdName(target);
      }

      if (auto error = FindCaseFallThrough(_, target_block, &case_fall_through,
                                           merge, case_targets, function)) {
        return error;
      }

      // Track how many time the fall through case has been targeted.
      if (case_fall_through != 0u) {
        auto where = num_fall_through_targeted.lower_bound(case_fall_through);
        if (where == num_fall_through_targeted.end() ||
            where->first != case_fall_through) {
          num_fall_through_targeted.insert(
              where, std::make_pair(case_fall_through, 1));
        } else {
          where->second++;
        }
      }
      seen_to_fall_through.insert(std::make_pair(target, case_fall_through));
    } else {
      case_fall_through = seen_iter->second;
    }

    if (case_fall_through == default_target &&
        !default_appears_multiple_times) {
      case_fall_through = default_case_fall_through;
    }
    if (case_fall_through != 0u) {
      bool is_default = i == 1;
      if (is_default) {
        default_case_fall_through = case_fall_through;
      } else {
        // Allow code like:
        // case x:
        // case y:
        //   ...
        // case z:
        //
        // Where x and y target the same block and fall through to z.
        uint32_t j = i;
        while ((j + 2 < switch_inst->operands().size()) &&
               target == switch_inst->GetOperandAs<uint32_t>(j + 2)) {
          j += 2;
        }
        // If Target T1 branches to Target T2, or if Target T1 branches to the
        // Default target and the Default target branches to Target T2, then T1
        // must immediately precede T2 in the list of OpSwitch Target operands.
        if ((switch_inst->operands().size() < j + 2) ||
            (case_fall_through != switch_inst->GetOperandAs<uint32_t>(j + 2))) {
          return _.diag(SPV_ERROR_INVALID_CFG, switch_inst)
                 << "Case construct that targets " << _.getIdName(target)
                 << " has branches to the case construct that targets "
                 << _.getIdName(case_fall_through)
                 << ", but does not immediately precede it in the "
                    "OpSwitch's target list";
        }
      }
    }
  }

  // Each case construct must be branched to by at most one other case
  // construct.
  for (const auto& pair : num_fall_through_targeted) {
    if (pair.second > 1) {
      return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef(pair.first))
             << "Multiple case constructs have branches to the case construct "
                "that targets "
             << _.getIdName(pair.first);
    }
  }

  return SPV_SUCCESS;
}

// Validates that all CFG divergences (i.e. conditional branch or switch) are
// structured correctly. Either divergence is preceded by a merge instruction
// or the divergence introduces at most one unseen label.
spv_result_t ValidateStructuredSelections(
    ValidationState_t& _, const std::vector<const BasicBlock*>& postorder) {
  std::unordered_set<uint32_t> seen;
  for (auto iter = postorder.rbegin(); iter != postorder.rend(); ++iter) {
    const auto* block = *iter;
    const auto* terminator = block->terminator();
    if (!terminator) continue;
    const auto index = terminator - &_.ordered_instructions()[0];
    auto* merge = &_.ordered_instructions()[index - 1];
    // Marks merges and continues as seen.
    if (merge->opcode() == spv::Op::OpSelectionMerge) {
      seen.insert(merge->GetOperandAs<uint32_t>(0));
    } else if (merge->opcode() == spv::Op::OpLoopMerge) {
      seen.insert(merge->GetOperandAs<uint32_t>(0));
      seen.insert(merge->GetOperandAs<uint32_t>(1));
    } else {
      // Only track the pointer if it is a merge instruction.
      merge = nullptr;
    }

    // Skip unreachable blocks.
    if (!block->structurally_reachable()) continue;

    if (terminator->opcode() == spv::Op::OpBranchConditional) {
      const auto true_label = terminator->GetOperandAs<uint32_t>(1);
      const auto false_label = terminator->GetOperandAs<uint32_t>(2);
      // Mark the upcoming blocks as seen now, but only error out if this block
      // was missing a merge instruction and both labels hadn't been seen
      // previously.
      const bool true_label_unseen = seen.insert(true_label).second;
      const bool false_label_unseen = seen.insert(false_label).second;
      if ((!merge || merge->opcode() == spv::Op::OpLoopMerge) &&
          true_label_unseen && false_label_unseen) {
        return _.diag(SPV_ERROR_INVALID_CFG, terminator)
               << "Selection must be structured";
      }
    } else if (terminator->opcode() == spv::Op::OpSwitch) {
      if (!merge) {
        return _.diag(SPV_ERROR_INVALID_CFG, terminator)
               << "OpSwitch must be preceded by an OpSelectionMerge "
                  "instruction";
      }
      // Mark the targets as seen.
      for (uint32_t i = 1; i < terminator->operands().size(); i += 2) {
        const auto target = terminator->GetOperandAs<uint32_t>(i);
        seen.insert(target);
      }
    }
  }

  return SPV_SUCCESS;
}

spv_result_t StructuredControlFlowChecks(
    ValidationState_t& _, Function* function,
    const std::vector<std::pair<uint32_t, uint32_t>>& back_edges,
    const std::vector<const BasicBlock*>& postorder) {
  /// Check all backedges target only loop headers and have exactly one
  /// back-edge branching to it

  // Map a loop header to blocks with back-edges to the loop header.
  std::map<uint32_t, std::unordered_set<uint32_t>> loop_latch_blocks;
  for (auto back_edge : back_edges) {
    uint32_t back_edge_block;
    uint32_t header_block;
    std::tie(back_edge_block, header_block) = back_edge;
    if (!function->IsBlockType(header_block, kBlockTypeLoop)) {
      return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef(back_edge_block))
             << "Back-edges (" << _.getIdName(back_edge_block) << " -> "
             << _.getIdName(header_block)
             << ") can only be formed between a block and a loop header.";
    }
    loop_latch_blocks[header_block].insert(back_edge_block);
  }

  // Check the loop headers have exactly one back-edge branching to it
  for (BasicBlock* loop_header : function->ordered_blocks()) {
    if (!loop_header->structurally_reachable()) continue;
    if (!loop_header->is_type(kBlockTypeLoop)) continue;
    auto loop_header_id = loop_header->id();
    auto num_latch_blocks = loop_latch_blocks[loop_header_id].size();
    if (num_latch_blocks != 1) {
      return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef(loop_header_id))
             << "Loop header " << _.getIdName(loop_header_id)
             << " is targeted by " << num_latch_blocks
             << " back-edge blocks but the standard requires exactly one";
    }
  }

  // Check construct rules
  for (const Construct& construct : function->constructs()) {
    auto header = construct.entry_block();
    if (!header->structurally_reachable()) continue;
    auto merge = construct.exit_block();

    if (!merge) {
      std::string construct_name, header_name, exit_name;
      std::tie(construct_name, header_name, exit_name) =
          ConstructNames(construct.type());
      return _.diag(SPV_ERROR_INTERNAL, _.FindDef(header->id()))
             << "Construct " + construct_name + " with " + header_name + " " +
                    _.getIdName(header->id()) + " does not have a " +
                    exit_name + ". This may be a bug in the validator.";
    }

    // If the header is reachable, the merge is guaranteed to be structurally
    // reachable.
    if (!header->structurally_dominates(*merge)) {
      return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef(merge->id()))
             << ConstructErrorString(construct, _.getIdName(header->id()),
                                     _.getIdName(merge->id()),
                                     "does not structurally dominate");
    }

    // If it's really a merge block for a selection or loop, then it must be
    // *strictly* structrually dominated by the header.
    if (construct.ExitBlockIsMergeBlock() && (header == merge)) {
      return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef(merge->id()))
             << ConstructErrorString(construct, _.getIdName(header->id()),
                                     _.getIdName(merge->id()),
                                     "does not strictly structurally dominate");
    }

    // Check post-dominance for continue constructs.  But dominance and
    // post-dominance only make sense when the construct is reachable.
    if (construct.type() == ConstructType::kContinue) {
      if (!merge->structurally_postdominates(*header)) {
        return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef(merge->id()))
               << ConstructErrorString(construct, _.getIdName(header->id()),
                                       _.getIdName(merge->id()),
                                       "is not structurally post dominated by");
      }
    }

    Construct::ConstructBlockSet construct_blocks = construct.blocks(function);
    std::string construct_name, header_name, exit_name;
    std::tie(construct_name, header_name, exit_name) =
        ConstructNames(construct.type());
    for (auto block : construct_blocks) {
      // Check that all exits from the construct are via structured exits.
      for (auto succ : *block->successors()) {
        if (!construct_blocks.count(succ) &&
            !construct.IsStructuredExit(_, succ)) {
          return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef(block->id()))
                 << "block <ID> " << _.getIdName(block->id()) << " exits the "
                 << construct_name << " headed by <ID> "
                 << _.getIdName(header->id())
                 << ", but not via a structured exit";
        }
      }
      if (block == header) continue;
      // Check that for all non-header blocks, all predecessors are within this
      // construct.
      for (auto pred : *block->predecessors()) {
        if (pred->structurally_reachable() && !construct_blocks.count(pred)) {
          return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef(pred->id()))
                 << "block <ID> " << pred->id() << " branches to the "
                 << construct_name << " construct, but not to the "
                 << header_name << " <ID> " << header->id();
        }
      }

      if (block->is_type(BlockType::kBlockTypeSelection) ||
          block->is_type(BlockType::kBlockTypeLoop)) {
        size_t index = (block->terminator() - &_.ordered_instructions()[0]) - 1;
        const auto& merge_inst = _.ordered_instructions()[index];
        if (merge_inst.opcode() == spv::Op::OpSelectionMerge ||
            merge_inst.opcode() == spv::Op::OpLoopMerge) {
          uint32_t merge_id = merge_inst.GetOperandAs<uint32_t>(0);
          auto merge_block = function->GetBlock(merge_id).first;
          if (merge_block->structurally_reachable() &&
              !construct_blocks.count(merge_block)) {
            return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef(block->id()))
                   << "Header block " << _.getIdName(block->id())
                   << " is contained in the " << construct_name
                   << " construct headed by " << _.getIdName(header->id())
                   << ", but its merge block " << _.getIdName(merge_id)
                   << " is not";
          }
        }
      }
    }

    if (construct.type() == ConstructType::kLoop) {
      // If the continue target differs from the loop header, then check that
      // all edges into the continue construct come from within the loop.
      const auto index = header->terminator() - &_.ordered_instructions()[0];
      const auto& merge_inst = _.ordered_instructions()[index - 1];
      const auto continue_id = merge_inst.GetOperandAs<uint32_t>(1);
      const auto* continue_inst = _.FindDef(continue_id);
      // OpLabel instructions aren't stored as part of the basic block for
      // legacy reaasons. Grab the next instruction and use it's block pointer
      // instead.
      const auto next_index =
          (continue_inst - &_.ordered_instructions()[0]) + 1;
      const auto& next_inst = _.ordered_instructions()[next_index];
      const auto* continue_target = next_inst.block();
      if (header->id() != continue_id) {
        for (auto pred : *continue_target->predecessors()) {
          // Ignore back-edges from within the continue construct.
          bool is_back_edge = false;
          for (auto back_edge : back_edges) {
            uint32_t back_edge_block;
            uint32_t header_block;
            std::tie(back_edge_block, header_block) = back_edge;
            if (header_block == continue_id && back_edge_block == pred->id())
              is_back_edge = true;
          }
          if (!construct_blocks.count(pred) && !is_back_edge) {
            return _.diag(SPV_ERROR_INVALID_CFG, pred->terminator())
                   << "Block " << _.getIdName(pred->id())
                   << " branches to the loop continue target "
                   << _.getIdName(continue_id)
                   << ", but is not contained in the associated loop construct "
                   << _.getIdName(header->id());
          }
        }
      }
    }

    // Checks rules for case constructs.
    if (construct.type() == ConstructType::kSelection &&
        header->terminator()->opcode() == spv::Op::OpSwitch) {
      const auto terminator = header->terminator();
      if (auto error =
              StructuredSwitchChecks(_, function, terminator, header, merge)) {
        return error;
      }
    }
  }

  if (auto error = ValidateStructuredSelections(_, postorder)) {
    return error;
  }

  return SPV_SUCCESS;
}

spv_result_t MaximalReconvergenceChecks(ValidationState_t& _) {
  // Find all the entry points with the MaximallyReconvergencesKHR execution
  // mode.
  std::unordered_set<uint32_t> maximal_funcs;
  std::unordered_set<uint32_t> maximal_entry_points;
  for (auto entry_point : _.entry_points()) {
    const auto* exec_modes = _.GetExecutionModes(entry_point);
    if (exec_modes &&
        exec_modes->count(spv::ExecutionMode::MaximallyReconvergesKHR)) {
      maximal_entry_points.insert(entry_point);
      maximal_funcs.insert(entry_point);
    }
  }

  if (maximal_entry_points.empty()) {
    return SPV_SUCCESS;
  }

  // Find all the functions reachable from a maximal reconvergence entry point.
  for (const auto& func : _.functions()) {
    const auto& entry_points = _.EntryPointReferences(func.id());
    for (auto id : entry_points) {
      if (maximal_entry_points.count(id)) {
        maximal_funcs.insert(func.id());
        break;
      }
    }
  }

  // Check for conditional branches with the same true and false targets.
  for (const auto& inst : _.ordered_instructions()) {
    if (inst.opcode() == spv::Op::OpBranchConditional) {
      const auto true_id = inst.GetOperandAs<uint32_t>(1);
      const auto false_id = inst.GetOperandAs<uint32_t>(2);
      if (true_id == false_id && maximal_funcs.count(inst.function()->id())) {
        return _.diag(SPV_ERROR_INVALID_ID, &inst)
               << "In entry points using the MaximallyReconvergesKHR execution "
                  "mode, True Label and False Label must be different labels";
      }
    }
  }

  // Check for invalid multiple predecessors. Only loop headers, continue
  // targets, merge targets or switch targets or defaults may have multiple
  // unique predecessors.
  for (const auto& func : _.functions()) {
    if (!maximal_funcs.count(func.id())) continue;

    for (const auto* block : func.ordered_blocks()) {
      std::unordered_set<uint32_t> unique_preds;
      const auto* preds = block->predecessors();
      if (!preds) continue;

      for (const auto* pred : *preds) {
        unique_preds.insert(pred->id());
      }
      if (unique_preds.size() < 2) continue;

      const auto* terminator = block->terminator();
      const auto index = terminator - &_.ordered_instructions()[0];
      const auto* pre_terminator = &_.ordered_instructions()[index - 1];
      if (pre_terminator->opcode() == spv::Op::OpLoopMerge) continue;

      const auto* label = _.FindDef(block->id());
      bool ok = false;
      for (const auto& pair : label->uses()) {
        const auto* use_inst = pair.first;
        switch (use_inst->opcode()) {
          case spv::Op::OpSelectionMerge:
          case spv::Op::OpLoopMerge:
          case spv::Op::OpSwitch:
            ok = true;
            break;
          default:
            break;
        }
      }
      if (!ok) {
        return _.diag(SPV_ERROR_INVALID_CFG, label)
               << "In entry points using the MaximallyReconvergesKHR "
                  "execution mode, this basic block must not have multiple "
                  "unique predecessors";
      }
    }
  }

  return SPV_SUCCESS;
}

spv_result_t PerformCfgChecks(ValidationState_t& _) {
  for (auto& function : _.functions()) {
    // Check all referenced blocks are defined within a function
    if (function.undefined_block_count() != 0) {
      std::string undef_blocks("{");
      bool first = true;
      for (auto undefined_block : function.undefined_blocks()) {
        undef_blocks += _.getIdName(undefined_block);
        if (!first) {
          undef_blocks += " ";
        }
        first = false;
      }
      return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef(function.id()))
             << "Block(s) " << undef_blocks << "}"
             << " are referenced but not defined in function "
             << _.getIdName(function.id());
    }

    // Set each block's immediate dominator.
    //
    // We want to analyze all the blocks in the function, even in degenerate
    // control flow cases including unreachable blocks.  So use the augmented
    // CFG to ensure we cover all the blocks.
    std::vector<const BasicBlock*> postorder;
    auto ignore_block = [](const BasicBlock*) {};
    auto no_terminal_blocks = [](const BasicBlock*) { return false; };
    if (!function.ordered_blocks().empty()) {
      /// calculate dominators
      CFA<BasicBlock>::DepthFirstTraversal(
          function.first_block(), function.AugmentedCFGSuccessorsFunction(),
          ignore_block, [&](const BasicBlock* b) { postorder.push_back(b); },
          no_terminal_blocks);
      auto edges = CFA<BasicBlock>::CalculateDominators(
          postorder, function.AugmentedCFGPredecessorsFunction());
      for (auto edge : edges) {
        if (edge.first != edge.second)
          edge.first->SetImmediateDominator(edge.second);
      }
    }

    auto& blocks = function.ordered_blocks();
    if (!blocks.empty()) {
      // Check if the order of blocks in the binary appear before the blocks
      // they dominate
      for (auto block = begin(blocks) + 1; block != end(blocks); ++block) {
        if (auto idom = (*block)->immediate_dominator()) {
          if (idom != function.pseudo_entry_block() &&
              block == std::find(begin(blocks), block, idom)) {
            return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef(idom->id()))
                   << "Block " << _.getIdName((*block)->id())
                   << " appears in the binary before its dominator "
                   << _.getIdName(idom->id());
          }
        }
      }
      // If we have structured control flow, check that no block has a control
      // flow nesting depth larger than the limit.
      if (_.HasCapability(spv::Capability::Shader)) {
        const int control_flow_nesting_depth_limit =
            _.options()->universal_limits_.max_control_flow_nesting_depth;
        for (auto block = begin(blocks); block != end(blocks); ++block) {
          if (function.GetBlockDepth(*block) >
              control_flow_nesting_depth_limit) {
            return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef((*block)->id()))
                   << "Maximum Control Flow nesting depth exceeded.";
          }
        }
      }
    }

    /// Structured control flow checks are only required for shader capabilities
    if (_.HasCapability(spv::Capability::Shader)) {
      // Calculate structural dominance.
      postorder.clear();
      std::vector<const BasicBlock*> postdom_postorder;
      std::vector<std::pair<uint32_t, uint32_t>> back_edges;
      if (!function.ordered_blocks().empty()) {
        /// calculate dominators
        CFA<BasicBlock>::DepthFirstTraversal(
            function.first_block(),
            function.AugmentedStructuralCFGSuccessorsFunction(), ignore_block,
            [&](const BasicBlock* b) { postorder.push_back(b); },
            no_terminal_blocks);
        auto edges = CFA<BasicBlock>::CalculateDominators(
            postorder, function.AugmentedStructuralCFGPredecessorsFunction());
        for (auto edge : edges) {
          if (edge.first != edge.second)
            edge.first->SetImmediateStructuralDominator(edge.second);
        }

        /// calculate post dominators
        CFA<BasicBlock>::DepthFirstTraversal(
            function.pseudo_exit_block(),
            function.AugmentedStructuralCFGPredecessorsFunction(), ignore_block,
            [&](const BasicBlock* b) { postdom_postorder.push_back(b); },
            no_terminal_blocks);
        auto postdom_edges = CFA<BasicBlock>::CalculateDominators(
            postdom_postorder,
            function.AugmentedStructuralCFGSuccessorsFunction());
        for (auto edge : postdom_edges) {
          edge.first->SetImmediateStructuralPostDominator(edge.second);
        }
        /// calculate back edges.
        CFA<BasicBlock>::DepthFirstTraversal(
            function.pseudo_entry_block(),
            function.AugmentedStructuralCFGSuccessorsFunction(), ignore_block,
            ignore_block,
            [&](const BasicBlock* from, const BasicBlock* to) {
              // A back edge must be a real edge. Since the augmented successors
              // contain structural edges, filter those from consideration.
              for (const auto* succ : *(from->successors())) {
                if (succ == to) back_edges.emplace_back(from->id(), to->id());
              }
            },
            no_terminal_blocks);
      }
      UpdateContinueConstructExitBlocks(function, back_edges);

      if (auto error =
              StructuredControlFlowChecks(_, &function, back_edges, postorder))
        return error;
    }
  }

  if (auto error = MaximalReconvergenceChecks(_)) {
    return error;
  }

  return SPV_SUCCESS;
}

spv_result_t CfgPass(ValidationState_t& _, const Instruction* inst) {
  spv::Op opcode = inst->opcode();
  switch (opcode) {
    case spv::Op::OpLabel:
      if (auto error = _.current_function().RegisterBlock(inst->id()))
        return error;

      // TODO(github:1661) This should be done in the
      // ValidationState::RegisterInstruction method but because of the order of
      // passes the OpLabel ends up not being part of the basic block it starts.
      _.current_function().current_block()->set_label(inst);
      break;
    case spv::Op::OpLoopMerge: {
      uint32_t merge_block = inst->GetOperandAs<uint32_t>(0);
      uint32_t continue_block = inst->GetOperandAs<uint32_t>(1);
      CFG_ASSERT(MergeBlockAssert, merge_block);

      if (auto error = _.current_function().RegisterLoopMerge(merge_block,
                                                              continue_block))
        return error;
    } break;
    case spv::Op::OpSelectionMerge: {
      uint32_t merge_block = inst->GetOperandAs<uint32_t>(0);
      CFG_ASSERT(MergeBlockAssert, merge_block);

      if (auto error = _.current_function().RegisterSelectionMerge(merge_block))
        return error;
    } break;
    case spv::Op::OpBranch: {
      uint32_t target = inst->GetOperandAs<uint32_t>(0);
      CFG_ASSERT(FirstBlockAssert, target);

      _.current_function().RegisterBlockEnd({target});
    } break;
    case spv::Op::OpBranchConditional: {
      uint32_t tlabel = inst->GetOperandAs<uint32_t>(1);
      uint32_t flabel = inst->GetOperandAs<uint32_t>(2);
      CFG_ASSERT(FirstBlockAssert, tlabel);
      CFG_ASSERT(FirstBlockAssert, flabel);

      _.current_function().RegisterBlockEnd({tlabel, flabel});
    } break;

    case spv::Op::OpSwitch: {
      std::vector<uint32_t> cases;
      for (size_t i = 1; i < inst->operands().size(); i += 2) {
        uint32_t target = inst->GetOperandAs<uint32_t>(i);
        CFG_ASSERT(FirstBlockAssert, target);
        cases.push_back(target);
      }
      _.current_function().RegisterBlockEnd({cases});
    } break;
    case spv::Op::OpReturn: {
      const uint32_t return_type = _.current_function().GetResultTypeId();
      const Instruction* return_type_inst = _.FindDef(return_type);
      assert(return_type_inst);
      if (return_type_inst->opcode() != spv::Op::OpTypeVoid)
        return _.diag(SPV_ERROR_INVALID_CFG, inst)
               << "OpReturn can only be called from a function with void "
               << "return type.";
      _.current_function().RegisterBlockEnd(std::vector<uint32_t>());
      break;
    }
    case spv::Op::OpKill:
    case spv::Op::OpReturnValue:
    case spv::Op::OpUnreachable:
    case spv::Op::OpTerminateInvocation:
    case spv::Op::OpIgnoreIntersectionKHR:
    case spv::Op::OpTerminateRayKHR:
    case spv::Op::OpEmitMeshTasksEXT:
      _.current_function().RegisterBlockEnd(std::vector<uint32_t>());
      // Ops with dedicated passes check for the Execution Model there
      if (opcode == spv::Op::OpKill) {
        _.current_function().RegisterExecutionModelLimitation(
            spv::ExecutionModel::Fragment,
            "OpKill requires Fragment execution model");
      }
      if (opcode == spv::Op::OpTerminateInvocation) {
        _.current_function().RegisterExecutionModelLimitation(
            spv::ExecutionModel::Fragment,
            "OpTerminateInvocation requires Fragment execution model");
      }
      if (opcode == spv::Op::OpIgnoreIntersectionKHR) {
        _.current_function().RegisterExecutionModelLimitation(
            spv::ExecutionModel::AnyHitKHR,
            "OpIgnoreIntersectionKHR requires AnyHitKHR execution model");
      }
      if (opcode == spv::Op::OpTerminateRayKHR) {
        _.current_function().RegisterExecutionModelLimitation(
            spv::ExecutionModel::AnyHitKHR,
            "OpTerminateRayKHR requires AnyHitKHR execution model");
      }

      break;
    default:
      break;
  }
  return SPV_SUCCESS;
}

void ReachabilityPass(ValidationState_t& _) {
  for (auto& f : _.functions()) {
    std::vector<BasicBlock*> stack;
    auto entry = f.first_block();
    // Skip function declarations.
    if (entry) stack.push_back(entry);

    while (!stack.empty()) {
      auto block = stack.back();
      stack.pop_back();

      if (block->reachable()) continue;

      block->set_reachable(true);
      for (auto succ : *block->successors()) {
        stack.push_back(succ);
      }
    }
  }

  // Repeat for structural reachability.
  for (auto& f : _.functions()) {
    std::vector<BasicBlock*> stack;
    auto entry = f.first_block();
    // Skip function declarations.
    if (entry) stack.push_back(entry);

    while (!stack.empty()) {
      auto block = stack.back();
      stack.pop_back();

      if (block->structurally_reachable()) continue;

      block->set_structurally_reachable(true);
      for (auto succ : *block->structural_successors()) {
        stack.push_back(succ);
      }
    }
  }
}

spv_result_t ControlFlowPass(ValidationState_t& _, const Instruction* inst) {
  switch (inst->opcode()) {
    case spv::Op::OpPhi:
      if (auto error = ValidatePhi(_, inst)) return error;
      break;
    case spv::Op::OpBranch:
      if (auto error = ValidateBranch(_, inst)) return error;
      break;
    case spv::Op::OpBranchConditional:
      if (auto error = ValidateBranchConditional(_, inst)) return error;
      break;
    case spv::Op::OpReturnValue:
      if (auto error = ValidateReturnValue(_, inst)) return error;
      break;
    case spv::Op::OpSwitch:
      if (auto error = ValidateSwitch(_, inst)) return error;
      break;
    case spv::Op::OpLoopMerge:
      if (auto error = ValidateLoopMerge(_, inst)) return error;
      break;
    default:
      break;
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
