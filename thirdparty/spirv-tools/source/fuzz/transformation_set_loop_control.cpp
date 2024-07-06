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

#include "source/fuzz/transformation_set_loop_control.h"

namespace spvtools {
namespace fuzz {

TransformationSetLoopControl::TransformationSetLoopControl(
    protobufs::TransformationSetLoopControl message)
    : message_(std::move(message)) {}

TransformationSetLoopControl::TransformationSetLoopControl(
    uint32_t block_id, uint32_t loop_control, uint32_t peel_count,
    uint32_t partial_count) {
  message_.set_block_id(block_id);
  message_.set_loop_control(loop_control);
  message_.set_peel_count(peel_count);
  message_.set_partial_count(partial_count);
}

bool TransformationSetLoopControl::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // |message_.block_id| must identify a block that ends with OpLoopMerge.
  auto block = ir_context->get_instr_block(message_.block_id());
  if (!block) {
    return false;
  }
  auto merge_inst = block->GetMergeInst();
  if (!merge_inst || merge_inst->opcode() != spv::Op::OpLoopMerge) {
    return false;
  }

  // We assert that the transformation does not try to set any meaningless bits
  // of the loop control mask.
  uint32_t all_loop_control_mask_bits_set = uint32_t(
      spv::LoopControlMask::Unroll | spv::LoopControlMask::DontUnroll |
      spv::LoopControlMask::DependencyInfinite |
      spv::LoopControlMask::DependencyLength |
      spv::LoopControlMask::MinIterations |
      spv::LoopControlMask::MaxIterations |
      spv::LoopControlMask::IterationMultiple |
      spv::LoopControlMask::PeelCount | spv::LoopControlMask::PartialCount);

  // The variable is only used in an assertion; the following keeps release-mode
  // compilers happy.
  (void)(all_loop_control_mask_bits_set);

  // No additional bits should be set.
  assert(!(message_.loop_control() & ~all_loop_control_mask_bits_set));

  // Grab the loop control mask currently associated with the OpLoopMerge
  // instruction.
  auto existing_loop_control_mask =
      merge_inst->GetSingleWordInOperand(kLoopControlMaskInOperandIndex);

  // Check that there is no attempt to set one of the loop controls that
  // requires guarantees to hold.
  for (spv::LoopControlMask mask : {spv::LoopControlMask::DependencyInfinite,
                                    spv::LoopControlMask::DependencyLength,
                                    spv::LoopControlMask::MinIterations,
                                    spv::LoopControlMask::MaxIterations,
                                    spv::LoopControlMask::IterationMultiple}) {
    // We have a problem if this loop control bit was not set in the original
    // loop control mask but is set by the transformation.
    if (LoopControlBitIsAddedByTransformation(mask,
                                              existing_loop_control_mask)) {
      return false;
    }
  }

  // Check that PeelCount and PartialCount are supported if used.
  if ((message_.loop_control() & uint32_t(spv::LoopControlMask::PeelCount)) &&
      !PeelCountIsSupported(ir_context)) {
    return false;
  }

  if ((message_.loop_control() &
       uint32_t(spv::LoopControlMask::PartialCount)) &&
      !PartialCountIsSupported(ir_context)) {
    return false;
  }

  if (message_.peel_count() > 0 &&
      !(message_.loop_control() & uint32_t(spv::LoopControlMask::PeelCount))) {
    // Peel count provided, but peel count mask bit not set.
    return false;
  }

  if (message_.partial_count() > 0 &&
      !(message_.loop_control() &
        uint32_t(spv::LoopControlMask::PartialCount))) {
    // Partial count provided, but partial count mask bit not set.
    return false;
  }

  // We must not set both 'don't unroll' and one of 'peel count' or 'partial
  // count'.
  return !(
      (message_.loop_control() & uint32_t(spv::LoopControlMask::DontUnroll)) &&
      (message_.loop_control() & uint32_t(spv::LoopControlMask::PeelCount |
                                          spv::LoopControlMask::PartialCount)));
}

void TransformationSetLoopControl::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  // Grab the loop merge instruction and its associated loop control mask.
  auto merge_inst =
      ir_context->get_instr_block(message_.block_id())->GetMergeInst();
  auto existing_loop_control_mask =
      merge_inst->GetSingleWordInOperand(kLoopControlMaskInOperandIndex);

  // We are going to replace the OpLoopMerge's operands with this list.
  opt::Instruction::OperandList new_operands;
  // We add the existing merge block and continue target ids.
  new_operands.push_back(merge_inst->GetInOperand(0));
  new_operands.push_back(merge_inst->GetInOperand(1));
  // We use the loop control mask from the transformation.
  new_operands.push_back(
      {SPV_OPERAND_TYPE_LOOP_CONTROL, {message_.loop_control()}});

  // It remains to determine what literals to provide, in association with
  // the new loop control mask.
  //
  // For the loop controls that require guarantees to hold about the number
  // of loop iterations, we need to keep, from the original OpLoopMerge, any
  // literals associated with loop control bits that are still set.

  uint32_t literal_index = 0;  // Indexes into the literals from the original
  // instruction.
  for (spv::LoopControlMask mask : {spv::LoopControlMask::DependencyLength,
                                    spv::LoopControlMask::MinIterations,
                                    spv::LoopControlMask::MaxIterations,
                                    spv::LoopControlMask::IterationMultiple}) {
    // Check whether the bit was set in the original loop control mask.
    if (existing_loop_control_mask & uint32_t(mask)) {
      // Check whether the bit is set in the new loop control mask.
      if (message_.loop_control() & uint32_t(mask)) {
        // Add the associated literal to our sequence of replacement operands.
        new_operands.push_back(
            {SPV_OPERAND_TYPE_LITERAL_INTEGER,
             {merge_inst->GetSingleWordInOperand(
                 kLoopControlFirstLiteralInOperandIndex + literal_index)}});
      }
      // Increment our index into the original loop control mask's literals,
      // whether or not the bit was set in the new mask.
      literal_index++;
    }
  }

  // If PeelCount is set in the new mask, |message_.peel_count| provides the
  // associated peel count.
  if (message_.loop_control() & uint32_t(spv::LoopControlMask::PeelCount)) {
    new_operands.push_back(
        {SPV_OPERAND_TYPE_LITERAL_INTEGER, {message_.peel_count()}});
  }

  // Similar, but for PartialCount.
  if (message_.loop_control() & uint32_t(spv::LoopControlMask::PartialCount)) {
    new_operands.push_back(
        {SPV_OPERAND_TYPE_LITERAL_INTEGER, {message_.partial_count()}});
  }

  // Replace the input operands of the OpLoopMerge with the new operands we have
  // accumulated.
  merge_inst->SetInOperands(std::move(new_operands));
}

protobufs::Transformation TransformationSetLoopControl::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_set_loop_control() = message_;
  return result;
}

bool TransformationSetLoopControl::LoopControlBitIsAddedByTransformation(
    spv::LoopControlMask loop_control_single_bit_mask,
    uint32_t existing_loop_control_mask) const {
  return !(uint32_t(loop_control_single_bit_mask) &
           existing_loop_control_mask) &&
         (uint32_t(loop_control_single_bit_mask) & message_.loop_control());
}

bool TransformationSetLoopControl::PartialCountIsSupported(
    opt::IRContext* ir_context) {
  // TODO(afd): We capture the environments for which this loop control is
  //  definitely not supported.  The check should be refined on demand for other
  //  target environments.
  switch (ir_context->grammar().target_env()) {
    case SPV_ENV_UNIVERSAL_1_0:
    case SPV_ENV_UNIVERSAL_1_1:
    case SPV_ENV_UNIVERSAL_1_2:
    case SPV_ENV_UNIVERSAL_1_3:
    case SPV_ENV_VULKAN_1_0:
    case SPV_ENV_VULKAN_1_1:
      return false;
    default:
      return true;
  }
}

bool TransformationSetLoopControl::PeelCountIsSupported(
    opt::IRContext* ir_context) {
  // TODO(afd): We capture the environments for which this loop control is
  //  definitely not supported.  The check should be refined on demand for other
  //  target environments.
  switch (ir_context->grammar().target_env()) {
    case SPV_ENV_UNIVERSAL_1_0:
    case SPV_ENV_UNIVERSAL_1_1:
    case SPV_ENV_UNIVERSAL_1_2:
    case SPV_ENV_UNIVERSAL_1_3:
    case SPV_ENV_VULKAN_1_0:
    case SPV_ENV_VULKAN_1_1:
      return false;
    default:
      return true;
  }
}

std::unordered_set<uint32_t> TransformationSetLoopControl::GetFreshIds() const {
  return std::unordered_set<uint32_t>();
}

}  // namespace fuzz
}  // namespace spvtools
