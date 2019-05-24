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

#include "source/val/validate.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "source/cfa.h"
#include "source/opcode.h"
#include "source/spirv_target_env.h"
#include "source/spirv_validator_options.h"
#include "source/val/basic_block.h"
#include "source/val/construct.h"
#include "source/val/function.h"
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

  const Instruction* type_inst = _.FindDef(inst->type_id());
  assert(type_inst);

  const SpvOp type_opcode = type_inst->opcode();
  if (type_opcode == SpvOpTypePointer &&
      _.addressing_model() == SpvAddressingModelLogical) {
    if (!_.features().variable_pointers &&
        !_.features().variable_pointers_storage_buffer) {
      return _.diag(SPV_ERROR_INVALID_DATA, inst)
             << "Using pointers with OpPhi requires capability "
             << "VariablePointers or VariablePointersStorageBuffer";
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
      if (_.GetIdOpcode(inc_id) != SpvOpLabel) {
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
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateBranch(ValidationState_t& _, const Instruction* inst) {
  // target operands must be OpLabel
  const auto id = inst->GetOperandAs<uint32_t>(0);
  const auto target = _.FindDef(id);
  if (!target || SpvOpLabel != target->opcode()) {
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
  if (!true_target || SpvOpLabel != true_target->opcode()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "The 'True Label' operand for OpBranchConditional must be the "
              "ID of an OpLabel instruction";
  }

  const auto false_id = inst->GetOperandAs<uint32_t>(2);
  const auto false_target = _.FindDef(false_id);
  if (!false_target || SpvOpLabel != false_target->opcode()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "The 'False Label' operand for OpBranchConditional must be the "
              "ID of an OpLabel instruction";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateSwitch(ValidationState_t& _, const Instruction* inst) {
  const auto num_operands = inst->operands().size();
  // At least two operands (selector, default), any more than that are
  // literal/target.

  // target operands must be OpLabel
  for (size_t i = 2; i < num_operands; i += 2) {
    // literal, id
    const auto id = inst->GetOperandAs<uint32_t>(i + 1);
    const auto target = _.FindDef(id);
    if (!target || SpvOpLabel != target->opcode()) {
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
           << "OpReturnValue Value <id> '" << _.getIdName(value_id)
           << "' does not represent a value.";
  }
  auto value_type = _.FindDef(value->type_id());
  if (!value_type || SpvOpTypeVoid == value_type->opcode()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpReturnValue value's type <id> '"
           << _.getIdName(value->type_id()) << "' is missing or void.";
  }

  const bool uses_variable_pointer =
      _.features().variable_pointers ||
      _.features().variable_pointers_storage_buffer;

  if (_.addressing_model() == SpvAddressingModelLogical &&
      SpvOpTypePointer == value_type->opcode() && !uses_variable_pointer &&
      !_.options()->relax_logical_pointer) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpReturnValue value's type <id> '"
           << _.getIdName(value->type_id())
           << "' is a pointer, which is invalid in the Logical addressing "
              "model.";
  }

  const auto function = inst->function();
  const auto return_type = _.FindDef(function->GetResultTypeId());
  if (!return_type || return_type->id() != value_type->id()) {
    return _.diag(SPV_ERROR_INVALID_ID, inst)
           << "OpReturnValue Value <id> '" << _.getIdName(value_id)
           << "'s type does not match OpFunction's return type.";
  }

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
  bool target_reachable = target_block->reachable();
  int target_depth = function->GetBlockDepth(target_block);
  while (!stack.empty()) {
    auto block = stack.back();
    stack.pop_back();

    if (block == merge) continue;

    if (!visited.insert(block).second) continue;

    if (target_reachable && block->reachable() &&
        target_block->dominates(*block)) {
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
  std::unordered_set<uint32_t> seen;
  for (uint32_t i = 1; i < switch_inst->operands().size(); i += 2) {
    uint32_t target = switch_inst->GetOperandAs<uint32_t>(i);
    if (target == merge->id()) continue;

    if (!seen.insert(target).second) continue;

    const auto target_block = function->GetBlock(target).first;
    // OpSwitch must dominate all its case constructs.
    if (header->reachable() && target_block->reachable() &&
        !header->dominates(*target_block)) {
      return _.diag(SPV_ERROR_INVALID_CFG, header->label())
             << "Selection header " << _.getIdName(header->id())
             << " does not dominate its case construct " << _.getIdName(target);
    }

    uint32_t case_fall_through = 0u;
    if (auto error = FindCaseFallThrough(_, target_block, &case_fall_through,
                                         merge, case_targets, function)) {
      return error;
    }

    // Track how many time the fall through case has been targeted.
    if (case_fall_through != 0u) {
      auto where = num_fall_through_targeted.lower_bound(case_fall_through);
      if (where == num_fall_through_targeted.end() ||
          where->first != case_fall_through) {
        num_fall_through_targeted.insert(where,
                                         std::make_pair(case_fall_through, 1));
      } else {
        where->second++;
      }
    }

    if (case_fall_through == default_target) {
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

spv_result_t StructuredControlFlowChecks(
    ValidationState_t& _, Function* function,
    const std::vector<std::pair<uint32_t, uint32_t>>& back_edges) {
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
    if (!loop_header->reachable()) continue;
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
    auto merge = construct.exit_block();

    if (header->reachable() && !merge) {
      std::string construct_name, header_name, exit_name;
      std::tie(construct_name, header_name, exit_name) =
          ConstructNames(construct.type());
      return _.diag(SPV_ERROR_INTERNAL, _.FindDef(header->id()))
             << "Construct " + construct_name + " with " + header_name + " " +
                    _.getIdName(header->id()) + " does not have a " +
                    exit_name + ". This may be a bug in the validator.";
    }

    // If the exit block is reachable then it's dominated by the
    // header.
    if (merge && merge->reachable()) {
      if (!header->dominates(*merge)) {
        return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef(merge->id()))
               << ConstructErrorString(construct, _.getIdName(header->id()),
                                       _.getIdName(merge->id()),
                                       "does not dominate");
      }
      // If it's really a merge block for a selection or loop, then it must be
      // *strictly* dominated by the header.
      if (construct.ExitBlockIsMergeBlock() && (header == merge)) {
        return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef(merge->id()))
               << ConstructErrorString(construct, _.getIdName(header->id()),
                                       _.getIdName(merge->id()),
                                       "does not strictly dominate");
      }
    }
    // Check post-dominance for continue constructs.  But dominance and
    // post-dominance only make sense when the construct is reachable.
    if (header->reachable() && construct.type() == ConstructType::kContinue) {
      if (!merge->postdominates(*header)) {
        return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef(merge->id()))
               << ConstructErrorString(construct, _.getIdName(header->id()),
                                       _.getIdName(merge->id()),
                                       "is not post dominated by");
      }
    }

    // Check that for all non-header blocks, all predecessors are within this
    // construct.
    Construct::ConstructBlockSet construct_blocks = construct.blocks(function);
    for (auto block : construct_blocks) {
      if (block == header) continue;
      for (auto pred : *block->predecessors()) {
        if (pred->reachable() && !construct_blocks.count(pred)) {
          std::string construct_name, header_name, exit_name;
          std::tie(construct_name, header_name, exit_name) =
              ConstructNames(construct.type());
          return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef(pred->id()))
                 << "block <ID> " << pred->id() << " branches to the "
                 << construct_name << " construct, but not to the "
                 << header_name << " <ID> " << header->id();
        }
      }
    }

    // Checks rules for case constructs.
    if (construct.type() == ConstructType::kSelection &&
        header->terminator()->opcode() == SpvOpSwitch) {
      const auto terminator = header->terminator();
      if (auto error =
              StructuredSwitchChecks(_, function, terminator, header, merge)) {
        return error;
      }
    }
  }
  return SPV_SUCCESS;
}

spv_result_t PerformWebGPUCfgChecks(ValidationState_t& _, Function* function) {
  for (auto& block : function->ordered_blocks()) {
    if (block->reachable()) continue;
    if (block->is_type(kBlockTypeMerge)) {
      // 1. Find the referencing merge and confirm that it is reachable.
      BasicBlock* merge_header = function->GetMergeHeader(block);
      assert(merge_header != nullptr);
      if (!merge_header->reachable()) {
        return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef(block->id()))
               << "For WebGPU, unreachable merge-blocks must be referenced by "
                  "a reachable merge instruction.";
      }

      // 2. Check that the only instructions are OpLabel and OpUnreachable.
      auto* label_inst = block->label();
      auto* terminator_inst = block->terminator();
      assert(label_inst != nullptr);
      assert(terminator_inst != nullptr);

      if (terminator_inst->opcode() != SpvOpUnreachable) {
        return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef(block->id()))
               << "For WebGPU, unreachable merge-blocks must terminate with "
                  "OpUnreachable.";
      }

      auto label_idx = label_inst - &_.ordered_instructions()[0];
      auto terminator_idx = terminator_inst - &_.ordered_instructions()[0];
      if (label_idx + 1 != terminator_idx) {
        return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef(block->id()))
               << "For WebGPU, unreachable merge-blocks must only contain an "
                  "OpLabel and OpUnreachable instruction.";
      }

      // 3. Use label instruction to confirm there is no uses by branches.
      for (auto use : label_inst->uses()) {
        const auto* use_inst = use.first;
        if (spvOpcodeIsBranch(use_inst->opcode())) {
          return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef(block->id()))
                 << "For WebGPU, unreachable merge-blocks cannot be the target "
                    "of a branch.";
        }
      }
    } else if (block->is_type(kBlockTypeContinue)) {
      // 1. Find referencing loop and confirm that it is reachable.
      std::vector<BasicBlock*> continue_headers =
          function->GetContinueHeaders(block);
      if (continue_headers.empty()) {
        return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef(block->id()))
               << "For WebGPU, unreachable continue-target must be referenced "
                  "by a loop instruction.";
      }

      std::vector<BasicBlock*> reachable_headers(continue_headers.size());
      auto iter =
          std::copy_if(continue_headers.begin(), continue_headers.end(),
                       reachable_headers.begin(),
                       [](BasicBlock* header) { return header->reachable(); });
      reachable_headers.resize(std::distance(reachable_headers.begin(), iter));

      if (reachable_headers.empty()) {
        return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef(block->id()))
               << "For WebGPU, unreachable continue-target must be referenced "
                  "by a reachable loop instruction.";
      }

      // 2. Check that the only instructions are OpLabel and OpBranch.
      auto* label_inst = block->label();
      auto* terminator_inst = block->terminator();
      assert(label_inst != nullptr);
      assert(terminator_inst != nullptr);

      if (terminator_inst->opcode() != SpvOpBranch) {
        return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef(block->id()))
               << "For WebGPU, unreachable continue-target must terminate with "
                  "OpBranch.";
      }

      auto label_idx = label_inst - &_.ordered_instructions()[0];
      auto terminator_idx = terminator_inst - &_.ordered_instructions()[0];
      if (label_idx + 1 != terminator_idx) {
        return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef(block->id()))
               << "For WebGPU, unreachable continue-target must only contain "
                  "an OpLabel and an OpBranch instruction.";
      }

      // 3. Use label instruction to confirm there is no uses by branches.
      for (auto use : label_inst->uses()) {
        const auto* use_inst = use.first;
        if (spvOpcodeIsBranch(use_inst->opcode())) {
          return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef(block->id()))
                 << "For WebGPU, unreachable continue-target cannot be the "
                    "target of a branch.";
        }
      }

      // 4. Confirm that continue-target has a back edge to a reachable loop
      //    header block.
      auto branch_target = terminator_inst->GetOperandAs<uint32_t>(0);
      for (auto* continue_header : reachable_headers) {
        if (branch_target != continue_header->id()) {
          return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef(block->id()))
                 << "For WebGPU, unreachable continue-target must only have a "
                    "back edge to a single reachable loop instruction.";
        }
      }
    } else {
      return _.diag(SPV_ERROR_INVALID_CFG, _.FindDef(block->id()))
             << "For WebGPU, all blocks must be reachable, unless they are "
             << "degenerate cases of merge-block or continue-target.";
    }
  }
  return SPV_SUCCESS;
}

// Checks that there are no OpUnreachable instructions reachable from the entry
// block of |function|.
spv_result_t ValidateUnreachableBlocks(ValidationState_t& _,
                                       Function& function) {
  auto* entry_block = function.first_block();
  std::vector<BasicBlock*> stack;
  std::unordered_set<BasicBlock*> seen;
  if (entry_block) stack.push_back(entry_block);

  while (!stack.empty()) {
    auto* block = stack.back();
    stack.pop_back();

    if (!seen.insert(block).second) continue;

    auto* terminator = block->terminator();
    if (terminator->opcode() == SpvOpUnreachable) {
      return _.diag(SPV_ERROR_INVALID_CFG, block->label())
             << "Statically reachable blocks cannot be terminated by "
                "OpUnreachable";
    }

    for (auto* succ : *block->successors()) {
      stack.push_back(succ);
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

    // Set each block's immediate dominator and immediate postdominator,
    // and find all back-edges.
    //
    // We want to analyze all the blocks in the function, even in degenerate
    // control flow cases including unreachable blocks.  So use the augmented
    // CFG to ensure we cover all the blocks.
    std::vector<const BasicBlock*> postorder;
    std::vector<const BasicBlock*> postdom_postorder;
    std::vector<std::pair<uint32_t, uint32_t>> back_edges;
    auto ignore_block = [](const BasicBlock*) {};
    auto ignore_edge = [](const BasicBlock*, const BasicBlock*) {};
    if (!function.ordered_blocks().empty()) {
      /// calculate dominators
      CFA<BasicBlock>::DepthFirstTraversal(
          function.first_block(), function.AugmentedCFGSuccessorsFunction(),
          ignore_block, [&](const BasicBlock* b) { postorder.push_back(b); },
          ignore_edge);
      auto edges = CFA<BasicBlock>::CalculateDominators(
          postorder, function.AugmentedCFGPredecessorsFunction());
      for (auto edge : edges) {
        edge.first->SetImmediateDominator(edge.second);
      }

      /// calculate post dominators
      CFA<BasicBlock>::DepthFirstTraversal(
          function.pseudo_exit_block(),
          function.AugmentedCFGPredecessorsFunction(), ignore_block,
          [&](const BasicBlock* b) { postdom_postorder.push_back(b); },
          ignore_edge);
      auto postdom_edges = CFA<BasicBlock>::CalculateDominators(
          postdom_postorder, function.AugmentedCFGSuccessorsFunction());
      for (auto edge : postdom_edges) {
        edge.first->SetImmediatePostDominator(edge.second);
      }
      /// calculate back edges.
      CFA<BasicBlock>::DepthFirstTraversal(
          function.pseudo_entry_block(),
          function
              .AugmentedCFGSuccessorsFunctionIncludingHeaderToContinueEdge(),
          ignore_block, ignore_block,
          [&](const BasicBlock* from, const BasicBlock* to) {
            back_edges.emplace_back(from->id(), to->id());
          });
    }
    UpdateContinueConstructExitBlocks(function, back_edges);

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

        // For WebGPU check that all unreachable blocks are degenerate cases for
        // merge-block or continue-target.
        if (spvIsWebGPUEnv(_.context()->target_env)) {
          spv_result_t result = PerformWebGPUCfgChecks(_, &function);
          if (result != SPV_SUCCESS) return result;
        }
      }
      // If we have structed control flow, check that no block has a control
      // flow nesting depth larger than the limit.
      if (_.HasCapability(SpvCapabilityShader)) {
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

    if (auto error = ValidateUnreachableBlocks(_, function)) return error;

    /// Structured control flow checks are only required for shader capabilities
    if (_.HasCapability(SpvCapabilityShader)) {
      if (auto error = StructuredControlFlowChecks(_, &function, back_edges))
        return error;
    }
  }
  return SPV_SUCCESS;
}

spv_result_t CfgPass(ValidationState_t& _, const Instruction* inst) {
  SpvOp opcode = inst->opcode();
  switch (opcode) {
    case SpvOpLabel:
      if (auto error = _.current_function().RegisterBlock(inst->id()))
        return error;

      // TODO(github:1661) This should be done in the
      // ValidationState::RegisterInstruction method but because of the order of
      // passes the OpLabel ends up not being part of the basic block it starts.
      _.current_function().current_block()->set_label(inst);
      break;
    case SpvOpLoopMerge: {
      uint32_t merge_block = inst->GetOperandAs<uint32_t>(0);
      uint32_t continue_block = inst->GetOperandAs<uint32_t>(1);
      CFG_ASSERT(MergeBlockAssert, merge_block);

      if (auto error = _.current_function().RegisterLoopMerge(merge_block,
                                                              continue_block))
        return error;
    } break;
    case SpvOpSelectionMerge: {
      uint32_t merge_block = inst->GetOperandAs<uint32_t>(0);
      CFG_ASSERT(MergeBlockAssert, merge_block);

      if (auto error = _.current_function().RegisterSelectionMerge(merge_block))
        return error;
    } break;
    case SpvOpBranch: {
      uint32_t target = inst->GetOperandAs<uint32_t>(0);
      CFG_ASSERT(FirstBlockAssert, target);

      _.current_function().RegisterBlockEnd({target}, opcode);
    } break;
    case SpvOpBranchConditional: {
      uint32_t tlabel = inst->GetOperandAs<uint32_t>(1);
      uint32_t flabel = inst->GetOperandAs<uint32_t>(2);
      CFG_ASSERT(FirstBlockAssert, tlabel);
      CFG_ASSERT(FirstBlockAssert, flabel);

      _.current_function().RegisterBlockEnd({tlabel, flabel}, opcode);
    } break;

    case SpvOpSwitch: {
      std::vector<uint32_t> cases;
      for (size_t i = 1; i < inst->operands().size(); i += 2) {
        uint32_t target = inst->GetOperandAs<uint32_t>(i);
        CFG_ASSERT(FirstBlockAssert, target);
        cases.push_back(target);
      }
      _.current_function().RegisterBlockEnd({cases}, opcode);
    } break;
    case SpvOpReturn: {
      const uint32_t return_type = _.current_function().GetResultTypeId();
      const Instruction* return_type_inst = _.FindDef(return_type);
      assert(return_type_inst);
      if (return_type_inst->opcode() != SpvOpTypeVoid)
        return _.diag(SPV_ERROR_INVALID_CFG, inst)
               << "OpReturn can only be called from a function with void "
               << "return type.";
    }
    // Fallthrough.
    case SpvOpKill:
    case SpvOpReturnValue:
    case SpvOpUnreachable:
      _.current_function().RegisterBlockEnd(std::vector<uint32_t>(), opcode);
      if (opcode == SpvOpKill) {
        _.current_function().RegisterExecutionModelLimitation(
            SpvExecutionModelFragment,
            "OpKill requires Fragment execution model");
      }
      break;
    default:
      break;
  }
  return SPV_SUCCESS;
}

spv_result_t ControlFlowPass(ValidationState_t& _, const Instruction* inst) {
  switch (inst->opcode()) {
    case SpvOpPhi:
      if (auto error = ValidatePhi(_, inst)) return error;
      break;
    case SpvOpBranch:
      if (auto error = ValidateBranch(_, inst)) return error;
      break;
    case SpvOpBranchConditional:
      if (auto error = ValidateBranchConditional(_, inst)) return error;
      break;
    case SpvOpReturnValue:
      if (auto error = ValidateReturnValue(_, inst)) return error;
      break;
    case SpvOpSwitch:
      if (auto error = ValidateSwitch(_, inst)) return error;
      break;
    default:
      break;
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
