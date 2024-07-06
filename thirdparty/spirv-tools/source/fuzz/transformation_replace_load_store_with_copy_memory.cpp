// Copyright (c) 2020 Google LLC
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

#include "transformation_replace_load_store_with_copy_memory.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/opcode.h"

namespace spvtools {
namespace fuzz {

namespace {
const uint32_t kOpStoreOperandIndexTargetVariable = 0;
const uint32_t kOpStoreOperandIndexIntermediateIdToWrite = 1;
const uint32_t kOpLoadOperandIndexSourceVariable = 2;
}  // namespace

TransformationReplaceLoadStoreWithCopyMemory::
    TransformationReplaceLoadStoreWithCopyMemory(
        protobufs::TransformationReplaceLoadStoreWithCopyMemory message)
    : message_(std::move(message)) {}

TransformationReplaceLoadStoreWithCopyMemory::
    TransformationReplaceLoadStoreWithCopyMemory(
        const protobufs::InstructionDescriptor& load_instruction_descriptor,
        const protobufs::InstructionDescriptor& store_instruction_descriptor) {
  *message_.mutable_load_instruction_descriptor() = load_instruction_descriptor;
  *message_.mutable_store_instruction_descriptor() =
      store_instruction_descriptor;
}
bool TransformationReplaceLoadStoreWithCopyMemory::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // This transformation is only applicable to the pair of OpLoad and OpStore
  // instructions.

  // The OpLoad instruction must be defined.
  auto load_instruction =
      FindInstruction(message_.load_instruction_descriptor(), ir_context);
  if (!load_instruction || load_instruction->opcode() != spv::Op::OpLoad) {
    return false;
  }

  // The OpStore instruction must be defined.
  auto store_instruction =
      FindInstruction(message_.store_instruction_descriptor(), ir_context);
  if (!store_instruction || store_instruction->opcode() != spv::Op::OpStore) {
    return false;
  }

  // Intermediate values of the OpLoad and the OpStore must match.
  if (load_instruction->result_id() !=
      store_instruction->GetSingleWordOperand(
          kOpStoreOperandIndexIntermediateIdToWrite)) {
    return false;
  }

  // Get storage class of the variable pointed by the source operand in OpLoad.
  opt::Instruction* source_id = ir_context->get_def_use_mgr()->GetDef(
      load_instruction->GetSingleWordOperand(2));
  spv::StorageClass storage_class = fuzzerutil::GetStorageClassFromPointerType(
      ir_context, source_id->type_id());

  // Iterate over all instructions between |load_instruction| and
  // |store_instruction|.
  for (auto it = load_instruction; it != store_instruction;
       it = it->NextNode()) {
    //|load_instruction| and |store_instruction| are not in the same block.
    if (it == nullptr) {
      return false;
    }

    // We need to make sure that the value pointed to by the source of the
    // OpLoad hasn't changed by the time we see the matching OpStore
    // instruction.
    if (IsMemoryWritingOpCode(it->opcode())) {
      return false;
    } else if (IsMemoryBarrierOpCode(it->opcode()) &&
               !IsStorageClassSafeAcrossMemoryBarriers(storage_class)) {
      return false;
    }
  }
  return true;
}

void TransformationReplaceLoadStoreWithCopyMemory::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  // OpLoad and OpStore instructions must be defined.
  auto load_instruction =
      FindInstruction(message_.load_instruction_descriptor(), ir_context);
  assert(load_instruction && load_instruction->opcode() == spv::Op::OpLoad &&
         "The required OpLoad instruction must be defined.");
  auto store_instruction =
      FindInstruction(message_.store_instruction_descriptor(), ir_context);
  assert(store_instruction && store_instruction->opcode() == spv::Op::OpStore &&
         "The required OpStore instruction must be defined.");

  // Intermediate values of the OpLoad and the OpStore must match.
  assert(load_instruction->result_id() ==
             store_instruction->GetSingleWordOperand(
                 kOpStoreOperandIndexIntermediateIdToWrite) &&
         "OpLoad and OpStore must refer to the same value.");

  // Get the ids of the source operand of the OpLoad and the target operand of
  // the OpStore.
  uint32_t source_variable_id =
      load_instruction->GetSingleWordOperand(kOpLoadOperandIndexSourceVariable);
  uint32_t target_variable_id = store_instruction->GetSingleWordOperand(
      kOpStoreOperandIndexTargetVariable);

  // Insert the OpCopyMemory instruction before the OpStore instruction.
  store_instruction->InsertBefore(MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpCopyMemory, 0, 0,
      opt::Instruction::OperandList(
          {{SPV_OPERAND_TYPE_ID, {target_variable_id}},
           {SPV_OPERAND_TYPE_ID, {source_variable_id}}})));

  // Remove the OpStore instruction.
  ir_context->KillInst(store_instruction);

  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

bool TransformationReplaceLoadStoreWithCopyMemory::IsMemoryWritingOpCode(
    spv::Op op_code) {
  if (spvOpcodeIsAtomicOp(op_code)) {
    return op_code != spv::Op::OpAtomicLoad;
  }
  switch (op_code) {
    case spv::Op::OpStore:
    case spv::Op::OpCopyMemory:
    case spv::Op::OpCopyMemorySized:
      return true;
    default:
      return false;
  }
}

bool TransformationReplaceLoadStoreWithCopyMemory::IsMemoryBarrierOpCode(
    spv::Op op_code) {
  switch (op_code) {
    case spv::Op::OpMemoryBarrier:
    case spv::Op::OpMemoryNamedBarrier:
      return true;
    default:
      return false;
  }
}

bool TransformationReplaceLoadStoreWithCopyMemory::
    IsStorageClassSafeAcrossMemoryBarriers(spv::StorageClass storage_class) {
  switch (storage_class) {
    case spv::StorageClass::UniformConstant:
    case spv::StorageClass::Input:
    case spv::StorageClass::Uniform:
    case spv::StorageClass::Private:
    case spv::StorageClass::Function:
      return true;
    default:
      return false;
  }
}

protobufs::Transformation
TransformationReplaceLoadStoreWithCopyMemory::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_replace_load_store_with_copy_memory() = message_;
  return result;
}

std::unordered_set<uint32_t>
TransformationReplaceLoadStoreWithCopyMemory::GetFreshIds() const {
  return std::unordered_set<uint32_t>();
}

}  // namespace fuzz
}  // namespace spvtools
