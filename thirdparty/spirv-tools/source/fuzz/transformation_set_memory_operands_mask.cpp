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

#include "source/fuzz/transformation_set_memory_operands_mask.h"

#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

namespace {

const uint32_t kOpLoadMemoryOperandsMaskIndex = 1;
const uint32_t kOpStoreMemoryOperandsMaskIndex = 2;
const uint32_t kOpCopyMemoryFirstMemoryOperandsMaskIndex = 2;
const uint32_t kOpCopyMemorySizedFirstMemoryOperandsMaskIndex = 3;

}  // namespace

TransformationSetMemoryOperandsMask::TransformationSetMemoryOperandsMask(
    protobufs::TransformationSetMemoryOperandsMask message)
    : message_(std::move(message)) {}

TransformationSetMemoryOperandsMask::TransformationSetMemoryOperandsMask(
    const protobufs::InstructionDescriptor& memory_access_instruction,
    uint32_t memory_operands_mask, uint32_t memory_operands_mask_index) {
  *message_.mutable_memory_access_instruction() = memory_access_instruction;
  message_.set_memory_operands_mask(memory_operands_mask);
  message_.set_memory_operands_mask_index(memory_operands_mask_index);
}

bool TransformationSetMemoryOperandsMask::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  if (message_.memory_operands_mask_index() != 0) {
    // The following conditions should never be violated, even if
    // transformations end up being replayed in a different way to the manner in
    // which they were applied during fuzzing, hence why these are assertions
    // rather than applicability checks.
    assert(message_.memory_operands_mask_index() == 1);
    assert(
        spv::Op(
            message_.memory_access_instruction().target_instruction_opcode()) ==
            spv::Op::OpCopyMemory ||
        spv::Op(
            message_.memory_access_instruction().target_instruction_opcode()) ==
            spv::Op::OpCopyMemorySized);
    assert(MultipleMemoryOperandMasksAreSupported(ir_context) &&
           "Multiple memory operand masks are not supported for this SPIR-V "
           "version.");
  }

  auto instruction =
      FindInstruction(message_.memory_access_instruction(), ir_context);
  if (!instruction) {
    return false;
  }
  if (!IsMemoryAccess(*instruction)) {
    return false;
  }

  auto original_mask_in_operand_index = GetInOperandIndexForMask(
      *instruction, message_.memory_operands_mask_index());
  assert(original_mask_in_operand_index != 0 &&
         "The given mask index is not valid.");
  uint32_t original_mask =
      original_mask_in_operand_index < instruction->NumInOperands()
          ? instruction->GetSingleWordInOperand(original_mask_in_operand_index)
          : static_cast<uint32_t>(spv::MemoryAccessMask::MaskNone);
  uint32_t new_mask = message_.memory_operands_mask();

  // Volatile must not be removed
  if ((original_mask & uint32_t(spv::MemoryAccessMask::Volatile)) &&
      !(new_mask & uint32_t(spv::MemoryAccessMask::Volatile))) {
    return false;
  }

  // Nontemporal can be added or removed, and no other flag is allowed to
  // change.  We do this by checking that the masks are equal once we set
  // their Volatile and Nontemporal flags to the same value (this works
  // because valid manipulation of Volatile is checked above, and the manner
  // in which Nontemporal is manipulated does not matter).
  return (original_mask | uint32_t(spv::MemoryAccessMask::Volatile) |
          uint32_t(spv::MemoryAccessMask::Nontemporal)) ==
         (new_mask | uint32_t(spv::MemoryAccessMask::Volatile) |
          uint32_t(spv::MemoryAccessMask::Nontemporal));
}

void TransformationSetMemoryOperandsMask::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  auto instruction =
      FindInstruction(message_.memory_access_instruction(), ir_context);
  auto original_mask_in_operand_index = GetInOperandIndexForMask(
      *instruction, message_.memory_operands_mask_index());
  // Either add a new operand, if no mask operand was already present, or
  // replace an existing mask operand.
  if (original_mask_in_operand_index >= instruction->NumInOperands()) {
    // Add first memory operand if it's missing.
    if (message_.memory_operands_mask_index() == 1 &&
        GetInOperandIndexForMask(*instruction, 0) >=
            instruction->NumInOperands()) {
      instruction->AddOperand({SPV_OPERAND_TYPE_MEMORY_ACCESS,
                               {uint32_t(spv::MemoryAccessMask::MaskNone)}});
    }

    instruction->AddOperand(
        {SPV_OPERAND_TYPE_MEMORY_ACCESS, {message_.memory_operands_mask()}});

  } else {
    instruction->SetInOperand(original_mask_in_operand_index,
                              {message_.memory_operands_mask()});
  }
}

protobufs::Transformation TransformationSetMemoryOperandsMask::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_set_memory_operands_mask() = message_;
  return result;
}

bool TransformationSetMemoryOperandsMask::IsMemoryAccess(
    const opt::Instruction& instruction) {
  switch (instruction.opcode()) {
    case spv::Op::OpLoad:
    case spv::Op::OpStore:
    case spv::Op::OpCopyMemory:
    case spv::Op::OpCopyMemorySized:
      return true;
    default:
      return false;
  }
}

uint32_t TransformationSetMemoryOperandsMask::GetInOperandIndexForMask(
    const opt::Instruction& instruction, uint32_t mask_index) {
  // Get the input operand index associated with the first memory operands mask
  // for the instruction.
  uint32_t first_mask_in_operand_index = 0;
  switch (instruction.opcode()) {
    case spv::Op::OpLoad:
      first_mask_in_operand_index = kOpLoadMemoryOperandsMaskIndex;
      break;
    case spv::Op::OpStore:
      first_mask_in_operand_index = kOpStoreMemoryOperandsMaskIndex;
      break;
    case spv::Op::OpCopyMemory:
      first_mask_in_operand_index = kOpCopyMemoryFirstMemoryOperandsMaskIndex;
      break;
    case spv::Op::OpCopyMemorySized:
      first_mask_in_operand_index =
          kOpCopyMemorySizedFirstMemoryOperandsMaskIndex;
      break;
    default:
      assert(false && "Unknown memory instruction.");
      break;
  }
  // If we are looking for the input operand index of the first mask, return it.
  // This will also return a correct value if the operand is missing.
  if (mask_index == 0) {
    return first_mask_in_operand_index;
  }
  assert(mask_index == 1 && "Memory operands mask index must be 0 or 1.");

  // Memory mask operands are optional. Thus, if the second operand exists,
  // its index will be >= |first_mask_in_operand_index + 1|. We can reason as
  // follows to separate the cases where the index of the second operand is
  // equal to |first_mask_in_operand_index + 1|:
  // - If the first memory operand doesn't exist, its value is equal to None.
  //   This means that it doesn't have additional operands following it and the
  //   condition in the if statement below will be satisfied.
  // - If the first memory operand exists and has no additional memory operands
  //   following it, the condition in the if statement below will be satisfied
  //   and we will return the correct value from the function.
  if (first_mask_in_operand_index + 1 >= instruction.NumInOperands()) {
    return first_mask_in_operand_index + 1;
  }

  // We are looking for the input operand index of the second mask.  This is a
  // little complicated because, depending on the contents of the first mask,
  // there may be some input operands separating the two masks.
  uint32_t first_mask =
      instruction.GetSingleWordInOperand(first_mask_in_operand_index);

  // Consider each bit that might have an associated extra input operand, and
  // count how many there are expected to be.
  uint32_t first_mask_extra_operand_count = 0;
  for (auto mask_bit : {spv::MemoryAccessMask::Aligned,
                        spv::MemoryAccessMask::MakePointerAvailable,
                        spv::MemoryAccessMask::MakePointerAvailableKHR,
                        spv::MemoryAccessMask::MakePointerVisible,
                        spv::MemoryAccessMask::MakePointerVisibleKHR}) {
    if (first_mask & uint32_t(mask_bit)) {
      first_mask_extra_operand_count++;
    }
  }
  return first_mask_in_operand_index + first_mask_extra_operand_count + 1;
}

bool TransformationSetMemoryOperandsMask::
    MultipleMemoryOperandMasksAreSupported(opt::IRContext* ir_context) {
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

std::unordered_set<uint32_t> TransformationSetMemoryOperandsMask::GetFreshIds()
    const {
  return std::unordered_set<uint32_t>();
}

}  // namespace fuzz
}  // namespace spvtools
