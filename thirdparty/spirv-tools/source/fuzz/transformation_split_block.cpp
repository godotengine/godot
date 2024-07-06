// Copyright (c) 2019 Google LLC
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

#include "source/fuzz/transformation_split_block.h"

#include <utility>

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace fuzz {

TransformationSplitBlock::TransformationSplitBlock(
    protobufs::TransformationSplitBlock message)
    : message_(std::move(message)) {}

TransformationSplitBlock::TransformationSplitBlock(
    const protobufs::InstructionDescriptor& instruction_to_split_before,
    uint32_t fresh_id) {
  *message_.mutable_instruction_to_split_before() = instruction_to_split_before;
  message_.set_fresh_id(fresh_id);
}

bool TransformationSplitBlock::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  if (!fuzzerutil::IsFreshId(ir_context, message_.fresh_id())) {
    // We require the id for the new block to be unused.
    return false;
  }
  auto instruction_to_split_before =
      FindInstruction(message_.instruction_to_split_before(), ir_context);
  if (!instruction_to_split_before) {
    // The instruction describing the block we should split does not exist.
    return false;
  }
  auto block_to_split =
      ir_context->get_instr_block(instruction_to_split_before);
  assert(block_to_split &&
         "We should not have managed to find the "
         "instruction if it was not contained in a block.");

  if (block_to_split->IsLoopHeader()) {
    // We cannot split a loop header block: back-edges would become invalid.
    return false;
  }

  auto split_before = fuzzerutil::GetIteratorForInstruction(
      block_to_split, instruction_to_split_before);
  assert(split_before != block_to_split->end() &&
         "At this point we know the"
         " block split point exists.");

  if (split_before->PreviousNode() &&
      split_before->PreviousNode()->opcode() == spv::Op::OpSelectionMerge) {
    // We cannot split directly after a selection merge: this would separate
    // the merge from its associated branch or switch operation.
    return false;
  }
  if (split_before->opcode() == spv::Op::OpVariable) {
    // We cannot split directly after a variable; variables in a function
    // must be contiguous in the entry block.
    return false;
  }
  // We cannot split before an OpPhi unless the OpPhi has exactly one
  // associated incoming edge.
  if (split_before->opcode() == spv::Op::OpPhi &&
      split_before->NumInOperands() != 2) {
    return false;
  }

  // Splitting the block must not separate the definition of an OpSampledImage
  // from its use: the SPIR-V data rules require them to be in the same block.
  return !fuzzerutil::
      SplittingBeforeInstructionSeparatesOpSampledImageDefinitionFromUse(
          block_to_split, instruction_to_split_before);
}

void TransformationSplitBlock::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  opt::Instruction* instruction_to_split_before =
      FindInstruction(message_.instruction_to_split_before(), ir_context);
  opt::BasicBlock* block_to_split =
      ir_context->get_instr_block(instruction_to_split_before);
  auto split_before = fuzzerutil::GetIteratorForInstruction(
      block_to_split, instruction_to_split_before);
  assert(split_before != block_to_split->end() &&
         "If the transformation is applicable, we should have an "
         "instruction to split on.");

  // We need to make sure the module's id bound is large enough to add the
  // fresh id.
  fuzzerutil::UpdateModuleIdBound(ir_context, message_.fresh_id());
  // Split the block.
  auto new_bb = block_to_split->SplitBasicBlock(ir_context, message_.fresh_id(),
                                                split_before);
  // The split does not automatically add a branch between the two parts of
  // the original block, so we add one.
  auto branch_instruction = MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpBranch, 0, 0,
      std::initializer_list<opt::Operand>{opt::Operand(
          spv_operand_type_t::SPV_OPERAND_TYPE_ID, {message_.fresh_id()})});
  auto branch_instruction_ptr = branch_instruction.get();
  block_to_split->AddInstruction(std::move(branch_instruction));

  // Inform the def-use manager about the branch instruction, and record its
  // block.
  ir_context->get_def_use_mgr()->AnalyzeInstDefUse(branch_instruction_ptr);
  ir_context->set_instr_block(branch_instruction_ptr, block_to_split);

  // If we split before OpPhi instructions, we need to update their
  // predecessor operand so that the block they used to be inside is now the
  // predecessor.
  new_bb->ForEachPhiInst([block_to_split,
                          ir_context](opt::Instruction* phi_inst) {
    assert(
        phi_inst->NumInOperands() == 2 &&
        "Precondition: a block can only be split before an OpPhi if the block"
        "has exactly one predecessor.");
    phi_inst->SetInOperand(1, {block_to_split->id()});
    ir_context->UpdateDefUse(phi_inst);
  });

  // We have updated the def-use manager and the instruction to block mapping,
  // but other analyses (especially control flow-related ones) need to be
  // recomputed.
  ir_context->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisDefUse |
      opt::IRContext::Analysis::kAnalysisInstrToBlockMapping);

  // If the block being split was dead, the new block arising from the split is
  // also dead.
  if (transformation_context->GetFactManager()->BlockIsDead(
          block_to_split->id())) {
    transformation_context->GetFactManager()->AddFactBlockIsDead(
        message_.fresh_id());
  }
}

protobufs::Transformation TransformationSplitBlock::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_split_block() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationSplitBlock::GetFreshIds() const {
  return {message_.fresh_id()};
}

}  // namespace fuzz
}  // namespace spvtools
