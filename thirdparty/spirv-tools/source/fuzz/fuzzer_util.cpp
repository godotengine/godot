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

#include "source/fuzz/fuzzer_util.h"

#include <algorithm>
#include <unordered_set>

#include "source/opt/build_module.h"

namespace spvtools {
namespace fuzz {

namespace fuzzerutil {
namespace {

// A utility class that uses RAII to change and restore the terminator
// instruction of the |block|.
class ChangeTerminatorRAII {
 public:
  explicit ChangeTerminatorRAII(opt::BasicBlock* block,
                                opt::Instruction new_terminator)
      : block_(block), old_terminator_(std::move(*block->terminator())) {
    *block_->terminator() = std::move(new_terminator);
  }

  ~ChangeTerminatorRAII() {
    *block_->terminator() = std::move(old_terminator_);
  }

 private:
  opt::BasicBlock* block_;
  opt::Instruction old_terminator_;
};

uint32_t MaybeGetOpConstant(opt::IRContext* ir_context,
                            const TransformationContext& transformation_context,
                            const std::vector<uint32_t>& words,
                            uint32_t type_id, bool is_irrelevant) {
  for (const auto& inst : ir_context->types_values()) {
    if (inst.opcode() == spv::Op::OpConstant && inst.type_id() == type_id &&
        inst.GetInOperand(0).words == words &&
        transformation_context.GetFactManager()->IdIsIrrelevant(
            inst.result_id()) == is_irrelevant) {
      return inst.result_id();
    }
  }

  return 0;
}

}  // namespace

const spvtools::MessageConsumer kSilentMessageConsumer =
    [](spv_message_level_t, const char*, const spv_position_t&,
       const char*) -> void {};

bool BuildIRContext(spv_target_env target_env,
                    const spvtools::MessageConsumer& message_consumer,
                    const std::vector<uint32_t>& binary_in,
                    spv_validator_options validator_options,
                    std::unique_ptr<spvtools::opt::IRContext>* ir_context) {
  SpirvTools tools(target_env);
  tools.SetMessageConsumer(message_consumer);
  if (!tools.IsValid()) {
    message_consumer(SPV_MSG_ERROR, nullptr, {},
                     "Failed to create SPIRV-Tools interface; stopping.");
    return false;
  }

  // Initial binary should be valid.
  if (!tools.Validate(binary_in.data(), binary_in.size(), validator_options)) {
    message_consumer(SPV_MSG_ERROR, nullptr, {},
                     "Initial binary is invalid; stopping.");
    return false;
  }

  // Build the module from the input binary.
  auto result = BuildModule(target_env, message_consumer, binary_in.data(),
                            binary_in.size());
  assert(result && "IRContext must be valid");
  *ir_context = std::move(result);
  return true;
}

bool IsFreshId(opt::IRContext* context, uint32_t id) {
  return !context->get_def_use_mgr()->GetDef(id);
}

void UpdateModuleIdBound(opt::IRContext* context, uint32_t id) {
  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/2541) consider the
  //  case where the maximum id bound is reached.
  context->module()->SetIdBound(
      std::max(context->module()->id_bound(), id + 1));
}

opt::BasicBlock* MaybeFindBlock(opt::IRContext* context,
                                uint32_t maybe_block_id) {
  auto inst = context->get_def_use_mgr()->GetDef(maybe_block_id);
  if (inst == nullptr) {
    // No instruction defining this id was found.
    return nullptr;
  }
  if (inst->opcode() != spv::Op::OpLabel) {
    // The instruction defining the id is not a label, so it cannot be a block
    // id.
    return nullptr;
  }
  return context->cfg()->block(maybe_block_id);
}

bool PhiIdsOkForNewEdge(
    opt::IRContext* context, opt::BasicBlock* bb_from, opt::BasicBlock* bb_to,
    const google::protobuf::RepeatedField<google::protobuf::uint32>& phi_ids) {
  if (bb_from->IsSuccessor(bb_to)) {
    // There is already an edge from |from_block| to |to_block|, so there is
    // no need to extend OpPhi instructions.  Do not allow phi ids to be
    // present. This might turn out to be too strict; perhaps it would be OK
    // just to ignore the ids in this case.
    return phi_ids.empty();
  }
  // The edge would add a previously non-existent edge from |from_block| to
  // |to_block|, so we go through the given phi ids and check that they exactly
  // match the OpPhi instructions in |to_block|.
  uint32_t phi_index = 0;
  // An explicit loop, rather than applying a lambda to each OpPhi in |bb_to|,
  // makes sense here because we need to increment |phi_index| for each OpPhi
  // instruction.
  for (auto& inst : *bb_to) {
    if (inst.opcode() != spv::Op::OpPhi) {
      // The OpPhi instructions all occur at the start of the block; if we find
      // a non-OpPhi then we have seen them all.
      break;
    }
    if (phi_index == static_cast<uint32_t>(phi_ids.size())) {
      // Not enough phi ids have been provided to account for the OpPhi
      // instructions.
      return false;
    }
    // Look for an instruction defining the next phi id.
    opt::Instruction* phi_extension =
        context->get_def_use_mgr()->GetDef(phi_ids[phi_index]);
    if (!phi_extension) {
      // The id given to extend this OpPhi does not exist.
      return false;
    }
    if (phi_extension->type_id() != inst.type_id()) {
      // The instruction given to extend this OpPhi either does not have a type
      // or its type does not match that of the OpPhi.
      return false;
    }

    if (context->get_instr_block(phi_extension)) {
      // The instruction defining the phi id has an associated block (i.e., it
      // is not a global value).  Check whether its definition dominates the
      // exit of |from_block|.
      auto dominator_analysis =
          context->GetDominatorAnalysis(bb_from->GetParent());
      if (!dominator_analysis->Dominates(phi_extension,
                                         bb_from->terminator())) {
        // The given id is no good as its definition does not dominate the exit
        // of |from_block|
        return false;
      }
    }
    phi_index++;
  }
  // We allow some of the ids provided for extending OpPhi instructions to be
  // unused.  Their presence does no harm, and requiring a perfect match may
  // make transformations less likely to cleanly apply.
  return true;
}

opt::Instruction CreateUnreachableEdgeInstruction(opt::IRContext* ir_context,
                                                  uint32_t bb_from_id,
                                                  uint32_t bb_to_id,
                                                  uint32_t bool_id) {
  const auto* bb_from = MaybeFindBlock(ir_context, bb_from_id);
  assert(bb_from && "|bb_from_id| is invalid");
  assert(MaybeFindBlock(ir_context, bb_to_id) && "|bb_to_id| is invalid");
  assert(bb_from->terminator()->opcode() == spv::Op::OpBranch &&
         "Precondition on terminator of bb_from is not satisfied");

  // Get the id of the boolean constant to be used as the condition.
  auto condition_inst = ir_context->get_def_use_mgr()->GetDef(bool_id);
  assert(condition_inst &&
         (condition_inst->opcode() == spv::Op::OpConstantTrue ||
          condition_inst->opcode() == spv::Op::OpConstantFalse) &&
         "|bool_id| is invalid");

  auto condition_value = condition_inst->opcode() == spv::Op::OpConstantTrue;
  auto successor_id = bb_from->terminator()->GetSingleWordInOperand(0);

  // Add the dead branch, by turning OpBranch into OpBranchConditional, and
  // ordering the targets depending on whether the given boolean corresponds to
  // true or false.
  return opt::Instruction(
      ir_context, spv::Op::OpBranchConditional, 0, 0,
      {{SPV_OPERAND_TYPE_ID, {bool_id}},
       {SPV_OPERAND_TYPE_ID, {condition_value ? successor_id : bb_to_id}},
       {SPV_OPERAND_TYPE_ID, {condition_value ? bb_to_id : successor_id}}});
}

void AddUnreachableEdgeAndUpdateOpPhis(
    opt::IRContext* context, opt::BasicBlock* bb_from, opt::BasicBlock* bb_to,
    uint32_t bool_id,
    const google::protobuf::RepeatedField<google::protobuf::uint32>& phi_ids) {
  assert(PhiIdsOkForNewEdge(context, bb_from, bb_to, phi_ids) &&
         "Precondition on phi_ids is not satisfied");

  const bool from_to_edge_already_exists = bb_from->IsSuccessor(bb_to);
  *bb_from->terminator() = CreateUnreachableEdgeInstruction(
      context, bb_from->id(), bb_to->id(), bool_id);

  // Update OpPhi instructions in the target block if this branch adds a
  // previously non-existent edge from source to target.
  if (!from_to_edge_already_exists) {
    uint32_t phi_index = 0;
    for (auto& inst : *bb_to) {
      if (inst.opcode() != spv::Op::OpPhi) {
        break;
      }
      assert(phi_index < static_cast<uint32_t>(phi_ids.size()) &&
             "There should be at least one phi id per OpPhi instruction.");
      inst.AddOperand({SPV_OPERAND_TYPE_ID, {phi_ids[phi_index]}});
      inst.AddOperand({SPV_OPERAND_TYPE_ID, {bb_from->id()}});
      phi_index++;
    }
  }
}

bool BlockIsBackEdge(opt::IRContext* context, uint32_t block_id,
                     uint32_t loop_header_id) {
  auto block = context->cfg()->block(block_id);
  auto loop_header = context->cfg()->block(loop_header_id);

  // |block| and |loop_header| must be defined, |loop_header| must be in fact
  // loop header and |block| must branch to it.
  if (!(block && loop_header && loop_header->IsLoopHeader() &&
        block->IsSuccessor(loop_header))) {
    return false;
  }

  // |block| must be reachable and be dominated by |loop_header|.
  opt::DominatorAnalysis* dominator_analysis =
      context->GetDominatorAnalysis(loop_header->GetParent());
  return context->IsReachable(*block) &&
         dominator_analysis->Dominates(loop_header, block);
}

bool BlockIsInLoopContinueConstruct(opt::IRContext* context, uint32_t block_id,
                                    uint32_t maybe_loop_header_id) {
  // We deem a block to be part of a loop's continue construct if the loop's
  // continue target dominates the block.
  auto containing_construct_block = context->cfg()->block(maybe_loop_header_id);
  if (containing_construct_block->IsLoopHeader()) {
    auto continue_target = containing_construct_block->ContinueBlockId();
    if (context->GetDominatorAnalysis(containing_construct_block->GetParent())
            ->Dominates(continue_target, block_id)) {
      return true;
    }
  }
  return false;
}

opt::BasicBlock::iterator GetIteratorForInstruction(
    opt::BasicBlock* block, const opt::Instruction* inst) {
  for (auto inst_it = block->begin(); inst_it != block->end(); ++inst_it) {
    if (inst == &*inst_it) {
      return inst_it;
    }
  }
  return block->end();
}

bool CanInsertOpcodeBeforeInstruction(
    spv::Op opcode, const opt::BasicBlock::iterator& instruction_in_block) {
  if (instruction_in_block->PreviousNode() &&
      (instruction_in_block->PreviousNode()->opcode() == spv::Op::OpLoopMerge ||
       instruction_in_block->PreviousNode()->opcode() ==
           spv::Op::OpSelectionMerge)) {
    // We cannot insert directly after a merge instruction.
    return false;
  }
  if (opcode != spv::Op::OpVariable &&
      instruction_in_block->opcode() == spv::Op::OpVariable) {
    // We cannot insert a non-OpVariable instruction directly before a
    // variable; variables in a function must be contiguous in the entry block.
    return false;
  }
  // We cannot insert a non-OpPhi instruction directly before an OpPhi, because
  // OpPhi instructions need to be contiguous at the start of a block.
  return opcode == spv::Op::OpPhi ||
         instruction_in_block->opcode() != spv::Op::OpPhi;
}

bool CanMakeSynonymOf(opt::IRContext* ir_context,
                      const TransformationContext& transformation_context,
                      const opt::Instruction& inst) {
  if (inst.opcode() == spv::Op::OpSampledImage) {
    // The SPIR-V data rules say that only very specific instructions may
    // may consume the result id of an OpSampledImage, and this excludes the
    // instructions that are used for making synonyms.
    return false;
  }
  if (!inst.HasResultId()) {
    // We can only make a synonym of an instruction that generates an id.
    return false;
  }
  if (transformation_context.GetFactManager()->IdIsIrrelevant(
          inst.result_id())) {
    // An irrelevant id can't be a synonym of anything.
    return false;
  }
  if (!inst.type_id()) {
    // We can only make a synonym of an instruction that has a type.
    return false;
  }
  auto type_inst = ir_context->get_def_use_mgr()->GetDef(inst.type_id());
  if (type_inst->opcode() == spv::Op::OpTypeVoid) {
    // We only make synonyms of instructions that define objects, and an object
    // cannot have void type.
    return false;
  }
  if (type_inst->opcode() == spv::Op::OpTypePointer) {
    switch (inst.opcode()) {
      case spv::Op::OpConstantNull:
      case spv::Op::OpUndef:
        // We disallow making synonyms of null or undefined pointers.  This is
        // to provide the property that if the original shader exhibited no bad
        // pointer accesses, the transformed shader will not either.
        return false;
      default:
        break;
    }
  }

  // We do not make synonyms of objects that have decorations: if the synonym is
  // not decorated analogously, using the original object vs. its synonymous
  // form may not be equivalent.
  return ir_context->get_decoration_mgr()
      ->GetDecorationsFor(inst.result_id(), true)
      .empty();
}

bool IsCompositeType(const opt::analysis::Type* type) {
  return type && (type->AsArray() || type->AsMatrix() || type->AsStruct() ||
                  type->AsVector());
}

std::vector<uint32_t> RepeatedFieldToVector(
    const google::protobuf::RepeatedField<uint32_t>& repeated_field) {
  std::vector<uint32_t> result;
  for (auto i : repeated_field) {
    result.push_back(i);
  }
  return result;
}

uint32_t WalkOneCompositeTypeIndex(opt::IRContext* context,
                                   uint32_t base_object_type_id,
                                   uint32_t index) {
  auto should_be_composite_type =
      context->get_def_use_mgr()->GetDef(base_object_type_id);
  assert(should_be_composite_type && "The type should exist.");
  switch (should_be_composite_type->opcode()) {
    case spv::Op::OpTypeArray: {
      auto array_length = GetArraySize(*should_be_composite_type, context);
      if (array_length == 0 || index >= array_length) {
        return 0;
      }
      return should_be_composite_type->GetSingleWordInOperand(0);
    }
    case spv::Op::OpTypeMatrix:
    case spv::Op::OpTypeVector: {
      auto count = should_be_composite_type->GetSingleWordInOperand(1);
      if (index >= count) {
        return 0;
      }
      return should_be_composite_type->GetSingleWordInOperand(0);
    }
    case spv::Op::OpTypeStruct: {
      if (index >= GetNumberOfStructMembers(*should_be_composite_type)) {
        return 0;
      }
      return should_be_composite_type->GetSingleWordInOperand(index);
    }
    default:
      return 0;
  }
}

uint32_t WalkCompositeTypeIndices(
    opt::IRContext* context, uint32_t base_object_type_id,
    const google::protobuf::RepeatedField<google::protobuf::uint32>& indices) {
  uint32_t sub_object_type_id = base_object_type_id;
  for (auto index : indices) {
    sub_object_type_id =
        WalkOneCompositeTypeIndex(context, sub_object_type_id, index);
    if (!sub_object_type_id) {
      return 0;
    }
  }
  return sub_object_type_id;
}

uint32_t GetNumberOfStructMembers(
    const opt::Instruction& struct_type_instruction) {
  assert(struct_type_instruction.opcode() == spv::Op::OpTypeStruct &&
         "An OpTypeStruct instruction is required here.");
  return struct_type_instruction.NumInOperands();
}

uint32_t GetArraySize(const opt::Instruction& array_type_instruction,
                      opt::IRContext* context) {
  auto array_length_constant =
      context->get_constant_mgr()
          ->GetConstantFromInst(context->get_def_use_mgr()->GetDef(
              array_type_instruction.GetSingleWordInOperand(1)))
          ->AsIntConstant();
  if (array_length_constant->words().size() != 1) {
    return 0;
  }
  return array_length_constant->GetU32();
}

uint32_t GetBoundForCompositeIndex(const opt::Instruction& composite_type_inst,
                                   opt::IRContext* ir_context) {
  switch (composite_type_inst.opcode()) {
    case spv::Op::OpTypeArray:
      return fuzzerutil::GetArraySize(composite_type_inst, ir_context);
    case spv::Op::OpTypeMatrix:
    case spv::Op::OpTypeVector:
      return composite_type_inst.GetSingleWordInOperand(1);
    case spv::Op::OpTypeStruct: {
      return fuzzerutil::GetNumberOfStructMembers(composite_type_inst);
    }
    case spv::Op::OpTypeRuntimeArray:
      assert(false &&
             "GetBoundForCompositeIndex should not be invoked with an "
             "OpTypeRuntimeArray, which does not have a static bound.");
      return 0;
    default:
      assert(false && "Unknown composite type.");
      return 0;
  }
}

spv::MemorySemanticsMask GetMemorySemanticsForStorageClass(
    spv::StorageClass storage_class) {
  switch (storage_class) {
    case spv::StorageClass::Workgroup:
      return spv::MemorySemanticsMask::WorkgroupMemory;

    case spv::StorageClass::StorageBuffer:
    case spv::StorageClass::PhysicalStorageBuffer:
      return spv::MemorySemanticsMask::UniformMemory;

    case spv::StorageClass::CrossWorkgroup:
      return spv::MemorySemanticsMask::CrossWorkgroupMemory;

    case spv::StorageClass::AtomicCounter:
      return spv::MemorySemanticsMask::AtomicCounterMemory;

    case spv::StorageClass::Image:
      return spv::MemorySemanticsMask::ImageMemory;

    default:
      return spv::MemorySemanticsMask::MaskNone;
  }
}

bool IsValid(const opt::IRContext* context,
             spv_validator_options validator_options,
             MessageConsumer consumer) {
  std::vector<uint32_t> binary;
  context->module()->ToBinary(&binary, false);
  SpirvTools tools(context->grammar().target_env());
  tools.SetMessageConsumer(std::move(consumer));
  return tools.Validate(binary.data(), binary.size(), validator_options);
}

bool IsValidAndWellFormed(const opt::IRContext* ir_context,
                          spv_validator_options validator_options,
                          MessageConsumer consumer) {
  if (!IsValid(ir_context, validator_options, consumer)) {
    // Expression to dump |ir_context| to /data/temp/shader.spv:
    //    DumpShader(ir_context, "/data/temp/shader.spv")
    consumer(SPV_MSG_INFO, nullptr, {},
             "Module is invalid (set a breakpoint to inspect).");
    return false;
  }
  // Check that all blocks in the module have appropriate parent functions.
  for (auto& function : *ir_context->module()) {
    for (auto& block : function) {
      if (block.GetParent() == nullptr) {
        std::stringstream ss;
        ss << "Block " << block.id() << " has no parent; its parent should be "
           << function.result_id() << " (set a breakpoint to inspect).";
        consumer(SPV_MSG_INFO, nullptr, {}, ss.str().c_str());
        return false;
      }
      if (block.GetParent() != &function) {
        std::stringstream ss;
        ss << "Block " << block.id() << " should have parent "
           << function.result_id() << " but instead has parent "
           << block.GetParent() << " (set a breakpoint to inspect).";
        consumer(SPV_MSG_INFO, nullptr, {}, ss.str().c_str());
        return false;
      }
    }
  }

  // Check that all instructions have distinct unique ids.  We map each unique
  // id to the first instruction it is observed to be associated with so that
  // if we encounter a duplicate we have access to the previous instruction -
  // this is a useful aid to debugging.
  std::unordered_map<uint32_t, opt::Instruction*> unique_ids;
  bool found_duplicate = false;
  ir_context->module()->ForEachInst([&consumer, &found_duplicate, ir_context,
                                     &unique_ids](opt::Instruction* inst) {
    (void)ir_context;  // Only used in an assertion; keep release-mode compilers
                       // happy.
    assert(inst->context() == ir_context &&
           "Instruction has wrong IR context.");
    if (unique_ids.count(inst->unique_id()) != 0) {
      consumer(SPV_MSG_INFO, nullptr, {},
               "Two instructions have the same unique id (set a breakpoint to "
               "inspect).");
      found_duplicate = true;
    }
    unique_ids.insert({inst->unique_id(), inst});
  });
  return !found_duplicate;
}

std::unique_ptr<opt::IRContext> CloneIRContext(opt::IRContext* context) {
  std::vector<uint32_t> binary;
  context->module()->ToBinary(&binary, false);
  return BuildModule(context->grammar().target_env(), nullptr, binary.data(),
                     binary.size());
}

bool IsNonFunctionTypeId(opt::IRContext* ir_context, uint32_t id) {
  auto type = ir_context->get_type_mgr()->GetType(id);
  return type && !type->AsFunction();
}

bool IsMergeOrContinue(opt::IRContext* ir_context, uint32_t block_id) {
  bool result = false;
  ir_context->get_def_use_mgr()->WhileEachUse(
      block_id,
      [&result](const opt::Instruction* use_instruction,
                uint32_t /*unused*/) -> bool {
        switch (use_instruction->opcode()) {
          case spv::Op::OpLoopMerge:
          case spv::Op::OpSelectionMerge:
            result = true;
            return false;
          default:
            return true;
        }
      });
  return result;
}

uint32_t GetLoopFromMergeBlock(opt::IRContext* ir_context,
                               uint32_t merge_block_id) {
  uint32_t result = 0;
  ir_context->get_def_use_mgr()->WhileEachUse(
      merge_block_id,
      [ir_context, &result](opt::Instruction* use_instruction,
                            uint32_t use_index) -> bool {
        switch (use_instruction->opcode()) {
          case spv::Op::OpLoopMerge:
            // The merge block operand is the first operand in OpLoopMerge.
            if (use_index == 0) {
              result = ir_context->get_instr_block(use_instruction)->id();
              return false;
            }
            return true;
          default:
            return true;
        }
      });
  return result;
}

uint32_t FindFunctionType(opt::IRContext* ir_context,
                          const std::vector<uint32_t>& type_ids) {
  // Look through the existing types for a match.
  for (auto& type_or_value : ir_context->types_values()) {
    if (type_or_value.opcode() != spv::Op::OpTypeFunction) {
      // We are only interested in function types.
      continue;
    }
    if (type_or_value.NumInOperands() != type_ids.size()) {
      // Not a match: different numbers of arguments.
      continue;
    }
    // Check whether the return type and argument types match.
    bool input_operands_match = true;
    for (uint32_t i = 0; i < type_or_value.NumInOperands(); i++) {
      if (type_ids[i] != type_or_value.GetSingleWordInOperand(i)) {
        input_operands_match = false;
        break;
      }
    }
    if (input_operands_match) {
      // Everything matches.
      return type_or_value.result_id();
    }
  }
  // No match was found.
  return 0;
}

opt::Instruction* GetFunctionType(opt::IRContext* context,
                                  const opt::Function* function) {
  uint32_t type_id = function->DefInst().GetSingleWordInOperand(1);
  return context->get_def_use_mgr()->GetDef(type_id);
}

opt::Function* FindFunction(opt::IRContext* ir_context, uint32_t function_id) {
  for (auto& function : *ir_context->module()) {
    if (function.result_id() == function_id) {
      return &function;
    }
  }
  return nullptr;
}

bool FunctionContainsOpKillOrUnreachable(const opt::Function& function) {
  for (auto& block : function) {
    if (block.terminator()->opcode() == spv::Op::OpKill ||
        block.terminator()->opcode() == spv::Op::OpUnreachable) {
      return true;
    }
  }
  return false;
}

bool FunctionIsEntryPoint(opt::IRContext* context, uint32_t function_id) {
  for (auto& entry_point : context->module()->entry_points()) {
    if (entry_point.GetSingleWordInOperand(1) == function_id) {
      return true;
    }
  }
  return false;
}

bool IdIsAvailableAtUse(opt::IRContext* context,
                        opt::Instruction* use_instruction,
                        uint32_t use_input_operand_index, uint32_t id) {
  assert(context->get_instr_block(use_instruction) &&
         "|use_instruction| must be in a basic block");

  auto defining_instruction = context->get_def_use_mgr()->GetDef(id);
  auto enclosing_function =
      context->get_instr_block(use_instruction)->GetParent();
  // If the id a function parameter, it needs to be associated with the
  // function containing the use.
  if (defining_instruction->opcode() == spv::Op::OpFunctionParameter) {
    return InstructionIsFunctionParameter(defining_instruction,
                                          enclosing_function);
  }
  if (!context->get_instr_block(id)) {
    // The id must be at global scope.
    return true;
  }
  if (defining_instruction == use_instruction) {
    // It is not OK for a definition to use itself.
    return false;
  }
  if (!context->IsReachable(*context->get_instr_block(use_instruction)) ||
      !context->IsReachable(*context->get_instr_block(id))) {
    // Skip unreachable blocks.
    return false;
  }
  auto dominator_analysis = context->GetDominatorAnalysis(enclosing_function);
  if (use_instruction->opcode() == spv::Op::OpPhi) {
    // In the case where the use is an operand to OpPhi, it is actually the
    // *parent* block associated with the operand that must be dominated by
    // the synonym.
    auto parent_block =
        use_instruction->GetSingleWordInOperand(use_input_operand_index + 1);
    return dominator_analysis->Dominates(
        context->get_instr_block(defining_instruction)->id(), parent_block);
  }
  return dominator_analysis->Dominates(defining_instruction, use_instruction);
}

bool IdIsAvailableBeforeInstruction(opt::IRContext* context,
                                    opt::Instruction* instruction,
                                    uint32_t id) {
  assert(context->get_instr_block(instruction) &&
         "|instruction| must be in a basic block");

  auto id_definition = context->get_def_use_mgr()->GetDef(id);
  auto function_enclosing_instruction =
      context->get_instr_block(instruction)->GetParent();
  // If the id a function parameter, it needs to be associated with the
  // function containing the instruction.
  if (id_definition->opcode() == spv::Op::OpFunctionParameter) {
    return InstructionIsFunctionParameter(id_definition,
                                          function_enclosing_instruction);
  }
  if (!context->get_instr_block(id)) {
    // The id is at global scope.
    return true;
  }
  if (id_definition == instruction) {
    // The instruction is not available right before its own definition.
    return false;
  }
  const auto* dominator_analysis =
      context->GetDominatorAnalysis(function_enclosing_instruction);
  if (context->IsReachable(*context->get_instr_block(instruction)) &&
      context->IsReachable(*context->get_instr_block(id)) &&
      dominator_analysis->Dominates(id_definition, instruction)) {
    // The id's definition dominates the instruction, and both the definition
    // and the instruction are in reachable blocks, thus the id is available at
    // the instruction.
    return true;
  }
  if (id_definition->opcode() == spv::Op::OpVariable &&
      function_enclosing_instruction ==
          context->get_instr_block(id)->GetParent()) {
    assert(!context->IsReachable(*context->get_instr_block(instruction)) &&
           "If the instruction were in a reachable block we should already "
           "have returned true.");
    // The id is a variable and it is in the same function as |instruction|.
    // This is OK despite |instruction| being unreachable.
    return true;
  }
  return false;
}

bool InstructionIsFunctionParameter(opt::Instruction* instruction,
                                    opt::Function* function) {
  if (instruction->opcode() != spv::Op::OpFunctionParameter) {
    return false;
  }
  bool found_parameter = false;
  function->ForEachParam(
      [instruction, &found_parameter](opt::Instruction* param) {
        if (param == instruction) {
          found_parameter = true;
        }
      });
  return found_parameter;
}

uint32_t GetTypeId(opt::IRContext* context, uint32_t result_id) {
  const auto* inst = context->get_def_use_mgr()->GetDef(result_id);
  assert(inst && "|result_id| is invalid");
  return inst->type_id();
}

uint32_t GetPointeeTypeIdFromPointerType(opt::Instruction* pointer_type_inst) {
  assert(pointer_type_inst &&
         pointer_type_inst->opcode() == spv::Op::OpTypePointer &&
         "Precondition: |pointer_type_inst| must be OpTypePointer.");
  return pointer_type_inst->GetSingleWordInOperand(1);
}

uint32_t GetPointeeTypeIdFromPointerType(opt::IRContext* context,
                                         uint32_t pointer_type_id) {
  return GetPointeeTypeIdFromPointerType(
      context->get_def_use_mgr()->GetDef(pointer_type_id));
}

spv::StorageClass GetStorageClassFromPointerType(
    opt::Instruction* pointer_type_inst) {
  assert(pointer_type_inst &&
         pointer_type_inst->opcode() == spv::Op::OpTypePointer &&
         "Precondition: |pointer_type_inst| must be OpTypePointer.");
  return static_cast<spv::StorageClass>(
      pointer_type_inst->GetSingleWordInOperand(0));
}

spv::StorageClass GetStorageClassFromPointerType(opt::IRContext* context,
                                                 uint32_t pointer_type_id) {
  return GetStorageClassFromPointerType(
      context->get_def_use_mgr()->GetDef(pointer_type_id));
}

uint32_t MaybeGetPointerType(opt::IRContext* context, uint32_t pointee_type_id,
                             spv::StorageClass storage_class) {
  for (auto& inst : context->types_values()) {
    switch (inst.opcode()) {
      case spv::Op::OpTypePointer:
        if (spv::StorageClass(inst.GetSingleWordInOperand(0)) ==
                storage_class &&
            inst.GetSingleWordInOperand(1) == pointee_type_id) {
          return inst.result_id();
        }
        break;
      default:
        break;
    }
  }
  return 0;
}

uint32_t InOperandIndexFromOperandIndex(const opt::Instruction& inst,
                                        uint32_t absolute_index) {
  // Subtract the number of non-input operands from the index
  return absolute_index - inst.NumOperands() + inst.NumInOperands();
}

bool IsNullConstantSupported(opt::IRContext* ir_context,
                             const opt::Instruction& type_inst) {
  switch (type_inst.opcode()) {
    case spv::Op::OpTypeArray:
    case spv::Op::OpTypeBool:
    case spv::Op::OpTypeDeviceEvent:
    case spv::Op::OpTypeEvent:
    case spv::Op::OpTypeFloat:
    case spv::Op::OpTypeInt:
    case spv::Op::OpTypeMatrix:
    case spv::Op::OpTypeQueue:
    case spv::Op::OpTypeReserveId:
    case spv::Op::OpTypeVector:
    case spv::Op::OpTypeStruct:
      return true;
    case spv::Op::OpTypePointer:
      // Null pointers are allowed if the VariablePointers capability is
      // enabled, or if the VariablePointersStorageBuffer capability is enabled
      // and the pointer type has StorageBuffer as its storage class.
      if (ir_context->get_feature_mgr()->HasCapability(
              spv::Capability::VariablePointers)) {
        return true;
      }
      if (ir_context->get_feature_mgr()->HasCapability(
              spv::Capability::VariablePointersStorageBuffer)) {
        return spv::StorageClass(type_inst.GetSingleWordInOperand(0)) ==
               spv::StorageClass::StorageBuffer;
      }
      return false;
    default:
      return false;
  }
}

bool GlobalVariablesMustBeDeclaredInEntryPointInterfaces(
    const opt::IRContext* ir_context) {
  // TODO(afd): We capture the environments for which this requirement holds.
  //  The check should be refined on demand for other target environments.
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

void AddVariableIdToEntryPointInterfaces(opt::IRContext* context, uint32_t id) {
  if (GlobalVariablesMustBeDeclaredInEntryPointInterfaces(context)) {
    // Conservatively add this global to the interface of every entry point in
    // the module.  This means that the global is available for other
    // transformations to use.
    //
    // A downside of this is that the global will be in the interface even if it
    // ends up never being used.
    //
    // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3111) revisit
    //  this if a more thorough approach to entry point interfaces is taken.
    for (auto& entry_point : context->module()->entry_points()) {
      entry_point.AddOperand({SPV_OPERAND_TYPE_ID, {id}});
    }
  }
}

opt::Instruction* AddGlobalVariable(opt::IRContext* context, uint32_t result_id,
                                    uint32_t type_id,
                                    spv::StorageClass storage_class,
                                    uint32_t initializer_id) {
  // Check various preconditions.
  assert(result_id != 0 && "Result id can't be 0");

  assert((storage_class == spv::StorageClass::Private ||
          storage_class == spv::StorageClass::Workgroup) &&
         "Variable's storage class must be either Private or Workgroup");

  auto* type_inst = context->get_def_use_mgr()->GetDef(type_id);
  (void)type_inst;  // Variable becomes unused in release mode.
  assert(type_inst && type_inst->opcode() == spv::Op::OpTypePointer &&
         GetStorageClassFromPointerType(type_inst) == storage_class &&
         "Variable's type is invalid");

  if (storage_class == spv::StorageClass::Workgroup) {
    assert(initializer_id == 0);
  }

  if (initializer_id != 0) {
    const auto* constant_inst =
        context->get_def_use_mgr()->GetDef(initializer_id);
    (void)constant_inst;  // Variable becomes unused in release mode.
    assert(constant_inst && spvOpcodeIsConstant(constant_inst->opcode()) &&
           GetPointeeTypeIdFromPointerType(type_inst) ==
               constant_inst->type_id() &&
           "Initializer is invalid");
  }

  opt::Instruction::OperandList operands = {
      {SPV_OPERAND_TYPE_STORAGE_CLASS, {static_cast<uint32_t>(storage_class)}}};

  if (initializer_id) {
    operands.push_back({SPV_OPERAND_TYPE_ID, {initializer_id}});
  }

  auto new_instruction = MakeUnique<opt::Instruction>(
      context, spv::Op::OpVariable, type_id, result_id, std::move(operands));
  auto result = new_instruction.get();
  context->module()->AddGlobalValue(std::move(new_instruction));

  AddVariableIdToEntryPointInterfaces(context, result_id);
  UpdateModuleIdBound(context, result_id);

  return result;
}

opt::Instruction* AddLocalVariable(opt::IRContext* context, uint32_t result_id,
                                   uint32_t type_id, uint32_t function_id,
                                   uint32_t initializer_id) {
  // Check various preconditions.
  assert(result_id != 0 && "Result id can't be 0");

  auto* type_inst = context->get_def_use_mgr()->GetDef(type_id);
  (void)type_inst;  // Variable becomes unused in release mode.
  assert(type_inst && type_inst->opcode() == spv::Op::OpTypePointer &&
         GetStorageClassFromPointerType(type_inst) ==
             spv::StorageClass::Function &&
         "Variable's type is invalid");

  const auto* constant_inst =
      context->get_def_use_mgr()->GetDef(initializer_id);
  (void)constant_inst;  // Variable becomes unused in release mode.
  assert(constant_inst && spvOpcodeIsConstant(constant_inst->opcode()) &&
         GetPointeeTypeIdFromPointerType(type_inst) ==
             constant_inst->type_id() &&
         "Initializer is invalid");

  auto* function = FindFunction(context, function_id);
  assert(function && "Function id is invalid");

  auto new_instruction = MakeUnique<opt::Instruction>(
      context, spv::Op::OpVariable, type_id, result_id,
      opt::Instruction::OperandList{{SPV_OPERAND_TYPE_STORAGE_CLASS,
                                     {uint32_t(spv::StorageClass::Function)}},
                                    {SPV_OPERAND_TYPE_ID, {initializer_id}}});
  auto result = new_instruction.get();
  function->begin()->begin()->InsertBefore(std::move(new_instruction));

  UpdateModuleIdBound(context, result_id);

  return result;
}

bool HasDuplicates(const std::vector<uint32_t>& arr) {
  return std::unordered_set<uint32_t>(arr.begin(), arr.end()).size() !=
         arr.size();
}

bool IsPermutationOfRange(const std::vector<uint32_t>& arr, uint32_t lo,
                          uint32_t hi) {
  if (arr.empty()) {
    return lo > hi;
  }

  if (HasDuplicates(arr)) {
    return false;
  }

  auto min_max = std::minmax_element(arr.begin(), arr.end());
  return arr.size() == hi - lo + 1 && *min_max.first == lo &&
         *min_max.second == hi;
}

std::vector<opt::Instruction*> GetParameters(opt::IRContext* ir_context,
                                             uint32_t function_id) {
  auto* function = FindFunction(ir_context, function_id);
  assert(function && "|function_id| is invalid");

  std::vector<opt::Instruction*> result;
  function->ForEachParam(
      [&result](opt::Instruction* inst) { result.push_back(inst); });

  return result;
}

void RemoveParameter(opt::IRContext* ir_context, uint32_t parameter_id) {
  auto* function = GetFunctionFromParameterId(ir_context, parameter_id);
  assert(function && "|parameter_id| is invalid");
  assert(!FunctionIsEntryPoint(ir_context, function->result_id()) &&
         "Can't remove parameter from an entry point function");

  function->RemoveParameter(parameter_id);

  // We've just removed parameters from the function and cleared their memory.
  // Make sure analyses have no dangling pointers.
  ir_context->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisNone);
}

std::vector<opt::Instruction*> GetCallers(opt::IRContext* ir_context,
                                          uint32_t function_id) {
  assert(FindFunction(ir_context, function_id) &&
         "|function_id| is not a result id of a function");

  std::vector<opt::Instruction*> result;
  ir_context->get_def_use_mgr()->ForEachUser(
      function_id, [&result, function_id](opt::Instruction* inst) {
        if (inst->opcode() == spv::Op::OpFunctionCall &&
            inst->GetSingleWordInOperand(0) == function_id) {
          result.push_back(inst);
        }
      });

  return result;
}

opt::Function* GetFunctionFromParameterId(opt::IRContext* ir_context,
                                          uint32_t param_id) {
  auto* param_inst = ir_context->get_def_use_mgr()->GetDef(param_id);
  assert(param_inst && "Parameter id is invalid");

  for (auto& function : *ir_context->module()) {
    if (InstructionIsFunctionParameter(param_inst, &function)) {
      return &function;
    }
  }

  return nullptr;
}

uint32_t UpdateFunctionType(opt::IRContext* ir_context, uint32_t function_id,
                            uint32_t new_function_type_result_id,
                            uint32_t return_type_id,
                            const std::vector<uint32_t>& parameter_type_ids) {
  // Check some initial constraints.
  assert(ir_context->get_type_mgr()->GetType(return_type_id) &&
         "Return type is invalid");
  for (auto id : parameter_type_ids) {
    const auto* type = ir_context->get_type_mgr()->GetType(id);
    (void)type;  // Make compilers happy in release mode.
    // Parameters can't be OpTypeVoid.
    assert(type && !type->AsVoid() && "Parameter has invalid type");
  }

  auto* function = FindFunction(ir_context, function_id);
  assert(function && "|function_id| is invalid");

  auto* old_function_type = GetFunctionType(ir_context, function);
  assert(old_function_type && "Function has invalid type");

  std::vector<uint32_t> operand_ids = {return_type_id};
  operand_ids.insert(operand_ids.end(), parameter_type_ids.begin(),
                     parameter_type_ids.end());

  // A trivial case - we change nothing.
  if (FindFunctionType(ir_context, operand_ids) ==
      old_function_type->result_id()) {
    return old_function_type->result_id();
  }

  if (ir_context->get_def_use_mgr()->NumUsers(old_function_type) == 1 &&
      FindFunctionType(ir_context, operand_ids) == 0) {
    // We can change |old_function_type| only if it's used once in the module
    // and we are certain we won't create a duplicate as a result of the change.

    // Update |old_function_type| in-place.
    opt::Instruction::OperandList operands;
    for (auto id : operand_ids) {
      operands.push_back({SPV_OPERAND_TYPE_ID, {id}});
    }

    old_function_type->SetInOperands(std::move(operands));

    // |operands| may depend on result ids defined below the |old_function_type|
    // in the module.
    old_function_type->RemoveFromList();
    ir_context->AddType(std::unique_ptr<opt::Instruction>(old_function_type));
    return old_function_type->result_id();
  } else {
    // We can't modify the |old_function_type| so we have to either use an
    // existing one or create a new one.
    auto type_id = FindOrCreateFunctionType(
        ir_context, new_function_type_result_id, operand_ids);
    assert(type_id != old_function_type->result_id() &&
           "We should've handled this case above");

    function->DefInst().SetInOperand(1, {type_id});

    // DefUseManager hasn't been updated yet, so if the following condition is
    // true, then |old_function_type| will have no users when this function
    // returns. We might as well remove it.
    if (ir_context->get_def_use_mgr()->NumUsers(old_function_type) == 1) {
      ir_context->KillInst(old_function_type);
    }

    return type_id;
  }
}

void AddFunctionType(opt::IRContext* ir_context, uint32_t result_id,
                     const std::vector<uint32_t>& type_ids) {
  assert(result_id != 0 && "Result id can't be 0");
  assert(!type_ids.empty() &&
         "OpTypeFunction always has at least one operand - function's return "
         "type");
  assert(IsNonFunctionTypeId(ir_context, type_ids[0]) &&
         "Return type must not be a function");

  for (size_t i = 1; i < type_ids.size(); ++i) {
    const auto* param_type = ir_context->get_type_mgr()->GetType(type_ids[i]);
    (void)param_type;  // Make compiler happy in release mode.
    assert(param_type && !param_type->AsVoid() && !param_type->AsFunction() &&
           "Function parameter can't have a function or void type");
  }

  opt::Instruction::OperandList operands;
  operands.reserve(type_ids.size());
  for (auto id : type_ids) {
    operands.push_back({SPV_OPERAND_TYPE_ID, {id}});
  }

  ir_context->AddType(MakeUnique<opt::Instruction>(
      ir_context, spv::Op::OpTypeFunction, 0, result_id, std::move(operands)));

  UpdateModuleIdBound(ir_context, result_id);
}

uint32_t FindOrCreateFunctionType(opt::IRContext* ir_context,
                                  uint32_t result_id,
                                  const std::vector<uint32_t>& type_ids) {
  if (auto existing_id = FindFunctionType(ir_context, type_ids)) {
    return existing_id;
  }
  AddFunctionType(ir_context, result_id, type_ids);
  return result_id;
}

uint32_t MaybeGetIntegerType(opt::IRContext* ir_context, uint32_t width,
                             bool is_signed) {
  opt::analysis::Integer type(width, is_signed);
  return ir_context->get_type_mgr()->GetId(&type);
}

uint32_t MaybeGetFloatType(opt::IRContext* ir_context, uint32_t width) {
  opt::analysis::Float type(width);
  return ir_context->get_type_mgr()->GetId(&type);
}

uint32_t MaybeGetBoolType(opt::IRContext* ir_context) {
  opt::analysis::Bool type;
  return ir_context->get_type_mgr()->GetId(&type);
}

uint32_t MaybeGetVectorType(opt::IRContext* ir_context,
                            uint32_t component_type_id,
                            uint32_t element_count) {
  const auto* component_type =
      ir_context->get_type_mgr()->GetType(component_type_id);
  assert(component_type &&
         (component_type->AsInteger() || component_type->AsFloat() ||
          component_type->AsBool()) &&
         "|component_type_id| is invalid");
  assert(element_count >= 2 && element_count <= 4 &&
         "Precondition: component count must be in range [2, 4].");
  opt::analysis::Vector type(component_type, element_count);
  return ir_context->get_type_mgr()->GetId(&type);
}

uint32_t MaybeGetStructType(opt::IRContext* ir_context,
                            const std::vector<uint32_t>& component_type_ids) {
  for (auto& type_or_value : ir_context->types_values()) {
    if (type_or_value.opcode() != spv::Op::OpTypeStruct ||
        type_or_value.NumInOperands() !=
            static_cast<uint32_t>(component_type_ids.size())) {
      continue;
    }
    bool all_components_match = true;
    for (uint32_t i = 0; i < component_type_ids.size(); i++) {
      if (type_or_value.GetSingleWordInOperand(i) != component_type_ids[i]) {
        all_components_match = false;
        break;
      }
    }
    if (all_components_match) {
      return type_or_value.result_id();
    }
  }
  return 0;
}

uint32_t MaybeGetVoidType(opt::IRContext* ir_context) {
  opt::analysis::Void type;
  return ir_context->get_type_mgr()->GetId(&type);
}

uint32_t MaybeGetZeroConstant(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context,
    uint32_t scalar_or_composite_type_id, bool is_irrelevant) {
  const auto* type_inst =
      ir_context->get_def_use_mgr()->GetDef(scalar_or_composite_type_id);
  assert(type_inst && "|scalar_or_composite_type_id| is invalid");

  switch (type_inst->opcode()) {
    case spv::Op::OpTypeBool:
      return MaybeGetBoolConstant(ir_context, transformation_context, false,
                                  is_irrelevant);
    case spv::Op::OpTypeFloat:
    case spv::Op::OpTypeInt: {
      const auto width = type_inst->GetSingleWordInOperand(0);
      std::vector<uint32_t> words = {0};
      if (width > 32) {
        words.push_back(0);
      }

      return MaybeGetScalarConstant(ir_context, transformation_context, words,
                                    scalar_or_composite_type_id, is_irrelevant);
    }
    case spv::Op::OpTypeStruct: {
      std::vector<uint32_t> component_ids;
      for (uint32_t i = 0; i < type_inst->NumInOperands(); ++i) {
        const auto component_type_id = type_inst->GetSingleWordInOperand(i);

        auto component_id =
            MaybeGetZeroConstant(ir_context, transformation_context,
                                 component_type_id, is_irrelevant);

        if (component_id == 0 && is_irrelevant) {
          // Irrelevant constants can use either relevant or irrelevant
          // constituents.
          component_id = MaybeGetZeroConstant(
              ir_context, transformation_context, component_type_id, false);
        }

        if (component_id == 0) {
          return 0;
        }

        component_ids.push_back(component_id);
      }

      return MaybeGetCompositeConstant(
          ir_context, transformation_context, component_ids,
          scalar_or_composite_type_id, is_irrelevant);
    }
    case spv::Op::OpTypeMatrix:
    case spv::Op::OpTypeVector: {
      const auto component_type_id = type_inst->GetSingleWordInOperand(0);

      auto component_id = MaybeGetZeroConstant(
          ir_context, transformation_context, component_type_id, is_irrelevant);

      if (component_id == 0 && is_irrelevant) {
        // Irrelevant constants can use either relevant or irrelevant
        // constituents.
        component_id = MaybeGetZeroConstant(ir_context, transformation_context,
                                            component_type_id, false);
      }

      if (component_id == 0) {
        return 0;
      }

      const auto component_count = type_inst->GetSingleWordInOperand(1);
      return MaybeGetCompositeConstant(
          ir_context, transformation_context,
          std::vector<uint32_t>(component_count, component_id),
          scalar_or_composite_type_id, is_irrelevant);
    }
    case spv::Op::OpTypeArray: {
      const auto component_type_id = type_inst->GetSingleWordInOperand(0);

      auto component_id = MaybeGetZeroConstant(
          ir_context, transformation_context, component_type_id, is_irrelevant);

      if (component_id == 0 && is_irrelevant) {
        // Irrelevant constants can use either relevant or irrelevant
        // constituents.
        component_id = MaybeGetZeroConstant(ir_context, transformation_context,
                                            component_type_id, false);
      }

      if (component_id == 0) {
        return 0;
      }

      return MaybeGetCompositeConstant(
          ir_context, transformation_context,
          std::vector<uint32_t>(GetArraySize(*type_inst, ir_context),
                                component_id),
          scalar_or_composite_type_id, is_irrelevant);
    }
    default:
      assert(false && "Type is not supported");
      return 0;
  }
}

bool CanCreateConstant(opt::IRContext* ir_context, uint32_t type_id) {
  opt::Instruction* type_instr = ir_context->get_def_use_mgr()->GetDef(type_id);
  assert(type_instr != nullptr && "The type must exist.");
  assert(spvOpcodeGeneratesType(type_instr->opcode()) &&
         "A type-generating opcode was expected.");
  switch (type_instr->opcode()) {
    case spv::Op::OpTypeBool:
    case spv::Op::OpTypeInt:
    case spv::Op::OpTypeFloat:
    case spv::Op::OpTypeMatrix:
    case spv::Op::OpTypeVector:
      return true;
    case spv::Op::OpTypeArray:
      return CanCreateConstant(ir_context,
                               type_instr->GetSingleWordInOperand(0));
    case spv::Op::OpTypeStruct:
      if (HasBlockOrBufferBlockDecoration(ir_context, type_id)) {
        return false;
      }
      for (uint32_t index = 0; index < type_instr->NumInOperands(); index++) {
        if (!CanCreateConstant(ir_context,
                               type_instr->GetSingleWordInOperand(index))) {
          return false;
        }
      }
      return true;
    default:
      return false;
  }
}

uint32_t MaybeGetScalarConstant(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context,
    const std::vector<uint32_t>& words, uint32_t scalar_type_id,
    bool is_irrelevant) {
  const auto* type = ir_context->get_type_mgr()->GetType(scalar_type_id);
  assert(type && "|scalar_type_id| is invalid");

  if (const auto* int_type = type->AsInteger()) {
    return MaybeGetIntegerConstant(ir_context, transformation_context, words,
                                   int_type->width(), int_type->IsSigned(),
                                   is_irrelevant);
  } else if (const auto* float_type = type->AsFloat()) {
    return MaybeGetFloatConstant(ir_context, transformation_context, words,
                                 float_type->width(), is_irrelevant);
  } else {
    assert(type->AsBool() && words.size() == 1 &&
           "|scalar_type_id| doesn't represent a scalar type");
    return MaybeGetBoolConstant(ir_context, transformation_context, words[0],
                                is_irrelevant);
  }
}

uint32_t MaybeGetCompositeConstant(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context,
    const std::vector<uint32_t>& component_ids, uint32_t composite_type_id,
    bool is_irrelevant) {
  const auto* type = ir_context->get_type_mgr()->GetType(composite_type_id);
  (void)type;  // Make compilers happy in release mode.
  assert(IsCompositeType(type) && "|composite_type_id| is invalid");

  for (const auto& inst : ir_context->types_values()) {
    if (inst.opcode() == spv::Op::OpConstantComposite &&
        inst.type_id() == composite_type_id &&
        transformation_context.GetFactManager()->IdIsIrrelevant(
            inst.result_id()) == is_irrelevant &&
        inst.NumInOperands() == component_ids.size()) {
      bool is_match = true;

      for (uint32_t i = 0; i < inst.NumInOperands(); ++i) {
        if (inst.GetSingleWordInOperand(i) != component_ids[i]) {
          is_match = false;
          break;
        }
      }

      if (is_match) {
        return inst.result_id();
      }
    }
  }

  return 0;
}

uint32_t MaybeGetIntegerConstant(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context,
    const std::vector<uint32_t>& words, uint32_t width, bool is_signed,
    bool is_irrelevant) {
  if (auto type_id = MaybeGetIntegerType(ir_context, width, is_signed)) {
    return MaybeGetOpConstant(ir_context, transformation_context, words,
                              type_id, is_irrelevant);
  }

  return 0;
}

uint32_t MaybeGetIntegerConstantFromValueAndType(opt::IRContext* ir_context,
                                                 uint32_t value,
                                                 uint32_t int_type_id) {
  auto int_type_inst = ir_context->get_def_use_mgr()->GetDef(int_type_id);

  assert(int_type_inst && "The given type id must exist.");

  auto int_type = ir_context->get_type_mgr()
                      ->GetType(int_type_inst->result_id())
                      ->AsInteger();

  assert(int_type && int_type->width() == 32 &&
         "The given type id must correspond to an 32-bit integer type.");

  opt::analysis::IntConstant constant(int_type, {value});

  // Check that the constant exists in the module.
  if (!ir_context->get_constant_mgr()->FindConstant(&constant)) {
    return 0;
  }

  return ir_context->get_constant_mgr()
      ->GetDefiningInstruction(&constant)
      ->result_id();
}

uint32_t MaybeGetFloatConstant(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context,
    const std::vector<uint32_t>& words, uint32_t width, bool is_irrelevant) {
  if (auto type_id = MaybeGetFloatType(ir_context, width)) {
    return MaybeGetOpConstant(ir_context, transformation_context, words,
                              type_id, is_irrelevant);
  }

  return 0;
}

uint32_t MaybeGetBoolConstant(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context, bool value,
    bool is_irrelevant) {
  if (auto type_id = MaybeGetBoolType(ir_context)) {
    for (const auto& inst : ir_context->types_values()) {
      if (inst.opcode() ==
              (value ? spv::Op::OpConstantTrue : spv::Op::OpConstantFalse) &&
          inst.type_id() == type_id &&
          transformation_context.GetFactManager()->IdIsIrrelevant(
              inst.result_id()) == is_irrelevant) {
        return inst.result_id();
      }
    }
  }

  return 0;
}

std::vector<uint32_t> IntToWords(uint64_t value, uint32_t width,
                                 bool is_signed) {
  assert(width <= 64 && "The bit width should not be more than 64 bits");

  // Sign-extend or zero-extend the last |width| bits of |value|, depending on
  // |is_signed|.
  if (is_signed) {
    // Sign-extend by shifting left and then shifting right, interpreting the
    // integer as signed.
    value = static_cast<int64_t>(value << (64 - width)) >> (64 - width);
  } else {
    // Zero-extend by shifting left and then shifting right, interpreting the
    // integer as unsigned.
    value = (value << (64 - width)) >> (64 - width);
  }

  std::vector<uint32_t> result;
  result.push_back(static_cast<uint32_t>(value));
  if (width > 32) {
    result.push_back(static_cast<uint32_t>(value >> 32));
  }
  return result;
}

bool TypesAreEqualUpToSign(opt::IRContext* ir_context, uint32_t type1_id,
                           uint32_t type2_id) {
  if (type1_id == type2_id) {
    return true;
  }

  auto type1 = ir_context->get_type_mgr()->GetType(type1_id);
  auto type2 = ir_context->get_type_mgr()->GetType(type2_id);

  // Integer scalar types must have the same width
  if (type1->AsInteger() && type2->AsInteger()) {
    return type1->AsInteger()->width() == type2->AsInteger()->width();
  }

  // Integer vector types must have the same number of components and their
  // component types must be integers with the same width.
  if (type1->AsVector() && type2->AsVector()) {
    auto component_type1 = type1->AsVector()->element_type()->AsInteger();
    auto component_type2 = type2->AsVector()->element_type()->AsInteger();

    // Only check the component count and width if they are integer.
    if (component_type1 && component_type2) {
      return type1->AsVector()->element_count() ==
                 type2->AsVector()->element_count() &&
             component_type1->width() == component_type2->width();
    }
  }

  // In all other cases, the types cannot be considered equal.
  return false;
}

std::map<uint32_t, uint32_t> RepeatedUInt32PairToMap(
    const google::protobuf::RepeatedPtrField<protobufs::UInt32Pair>& data) {
  std::map<uint32_t, uint32_t> result;

  for (const auto& entry : data) {
    result[entry.first()] = entry.second();
  }

  return result;
}

google::protobuf::RepeatedPtrField<protobufs::UInt32Pair>
MapToRepeatedUInt32Pair(const std::map<uint32_t, uint32_t>& data) {
  google::protobuf::RepeatedPtrField<protobufs::UInt32Pair> result;

  for (const auto& entry : data) {
    protobufs::UInt32Pair pair;
    pair.set_first(entry.first);
    pair.set_second(entry.second);
    *result.Add() = std::move(pair);
  }

  return result;
}

opt::Instruction* GetLastInsertBeforeInstruction(opt::IRContext* ir_context,
                                                 uint32_t block_id,
                                                 spv::Op opcode) {
  // CFG::block uses std::map::at which throws an exception when |block_id| is
  // invalid. The error message is unhelpful, though. Thus, we test that
  // |block_id| is valid here.
  const auto* label_inst = ir_context->get_def_use_mgr()->GetDef(block_id);
  (void)label_inst;  // Make compilers happy in release mode.
  assert(label_inst && label_inst->opcode() == spv::Op::OpLabel &&
         "|block_id| is invalid");

  auto* block = ir_context->cfg()->block(block_id);
  auto it = block->rbegin();
  assert(it != block->rend() && "Basic block can't be empty");

  if (block->GetMergeInst()) {
    ++it;
    assert(it != block->rend() &&
           "|block| must have at least two instructions:"
           "terminator and a merge instruction");
  }

  return CanInsertOpcodeBeforeInstruction(opcode, &*it) ? &*it : nullptr;
}

bool IdUseCanBeReplaced(opt::IRContext* ir_context,
                        const TransformationContext& transformation_context,
                        opt::Instruction* use_instruction,
                        uint32_t use_in_operand_index) {
  if (spvOpcodeIsAccessChain(use_instruction->opcode()) &&
      use_in_operand_index > 0) {
    // A replacement for an irrelevant index in OpAccessChain must be clamped
    // first.
    if (transformation_context.GetFactManager()->IdIsIrrelevant(
            use_instruction->GetSingleWordInOperand(use_in_operand_index))) {
      return false;
    }

    // This is an access chain index.  If the (sub-)object being accessed by the
    // given index has struct type then we cannot replace the use, as it needs
    // to be an OpConstant.

    // Get the top-level composite type that is being accessed.
    auto object_being_accessed = ir_context->get_def_use_mgr()->GetDef(
        use_instruction->GetSingleWordInOperand(0));
    auto pointer_type =
        ir_context->get_type_mgr()->GetType(object_being_accessed->type_id());
    assert(pointer_type->AsPointer());
    auto composite_type_being_accessed =
        pointer_type->AsPointer()->pointee_type();

    // Now walk the access chain, tracking the type of each sub-object of the
    // composite that is traversed, until the index of interest is reached.
    for (uint32_t index_in_operand = 1; index_in_operand < use_in_operand_index;
         index_in_operand++) {
      // For vectors, matrices and arrays, getting the type of the sub-object is
      // trivial. For the struct case, the sub-object type is field-sensitive,
      // and depends on the constant index that is used.
      if (composite_type_being_accessed->AsVector()) {
        composite_type_being_accessed =
            composite_type_being_accessed->AsVector()->element_type();
      } else if (composite_type_being_accessed->AsMatrix()) {
        composite_type_being_accessed =
            composite_type_being_accessed->AsMatrix()->element_type();
      } else if (composite_type_being_accessed->AsArray()) {
        composite_type_being_accessed =
            composite_type_being_accessed->AsArray()->element_type();
      } else if (composite_type_being_accessed->AsRuntimeArray()) {
        composite_type_being_accessed =
            composite_type_being_accessed->AsRuntimeArray()->element_type();
      } else {
        assert(composite_type_being_accessed->AsStruct());
        auto constant_index_instruction = ir_context->get_def_use_mgr()->GetDef(
            use_instruction->GetSingleWordInOperand(index_in_operand));
        assert(constant_index_instruction->opcode() == spv::Op::OpConstant);
        uint32_t member_index =
            constant_index_instruction->GetSingleWordInOperand(0);
        composite_type_being_accessed =
            composite_type_being_accessed->AsStruct()
                ->element_types()[member_index];
      }
    }

    // We have found the composite type being accessed by the index we are
    // considering replacing. If it is a struct, then we cannot do the
    // replacement as struct indices must be constants.
    if (composite_type_being_accessed->AsStruct()) {
      return false;
    }
  }

  if (use_instruction->opcode() == spv::Op::OpFunctionCall &&
      use_in_operand_index > 0) {
    // This is a function call argument.  It is not allowed to have pointer
    // type.

    // Get the definition of the function being called.
    auto function = ir_context->get_def_use_mgr()->GetDef(
        use_instruction->GetSingleWordInOperand(0));
    // From the function definition, get the function type.
    auto function_type = ir_context->get_def_use_mgr()->GetDef(
        function->GetSingleWordInOperand(1));
    // OpTypeFunction's 0-th input operand is the function return type, and the
    // function argument types follow. Because the arguments to OpFunctionCall
    // start from input operand 1, we can use |use_in_operand_index| to get the
    // type associated with this function argument.
    auto parameter_type = ir_context->get_type_mgr()->GetType(
        function_type->GetSingleWordInOperand(use_in_operand_index));
    if (parameter_type->AsPointer()) {
      return false;
    }
  }

  if (use_instruction->opcode() == spv::Op::OpImageTexelPointer &&
      use_in_operand_index == 2) {
    // The OpImageTexelPointer instruction has a Sample parameter that in some
    // situations must be an id for the value 0.  To guard against disrupting
    // that requirement, we do not replace this argument to that instruction.
    return false;
  }

  if (ir_context->get_feature_mgr()->HasCapability(spv::Capability::Shader)) {
    // With the Shader capability, memory scope and memory semantics operands
    // are required to be constants, so they cannot be replaced arbitrarily.
    switch (use_instruction->opcode()) {
      case spv::Op::OpAtomicLoad:
      case spv::Op::OpAtomicStore:
      case spv::Op::OpAtomicExchange:
      case spv::Op::OpAtomicIIncrement:
      case spv::Op::OpAtomicIDecrement:
      case spv::Op::OpAtomicIAdd:
      case spv::Op::OpAtomicISub:
      case spv::Op::OpAtomicSMin:
      case spv::Op::OpAtomicUMin:
      case spv::Op::OpAtomicSMax:
      case spv::Op::OpAtomicUMax:
      case spv::Op::OpAtomicAnd:
      case spv::Op::OpAtomicOr:
      case spv::Op::OpAtomicXor:
        if (use_in_operand_index == 1 || use_in_operand_index == 2) {
          return false;
        }
        break;
      case spv::Op::OpAtomicCompareExchange:
        if (use_in_operand_index == 1 || use_in_operand_index == 2 ||
            use_in_operand_index == 3) {
          return false;
        }
        break;
      case spv::Op::OpAtomicCompareExchangeWeak:
      case spv::Op::OpAtomicFlagTestAndSet:
      case spv::Op::OpAtomicFlagClear:
      case spv::Op::OpAtomicFAddEXT:
        assert(false && "Not allowed with the Shader capability.");
      default:
        break;
    }
  }

  return true;
}

bool MembersHaveBuiltInDecoration(opt::IRContext* ir_context,
                                  uint32_t struct_type_id) {
  const auto* type_inst = ir_context->get_def_use_mgr()->GetDef(struct_type_id);
  assert(type_inst && type_inst->opcode() == spv::Op::OpTypeStruct &&
         "|struct_type_id| is not a result id of an OpTypeStruct");

  uint32_t builtin_count = 0;
  ir_context->get_def_use_mgr()->ForEachUser(
      type_inst,
      [struct_type_id, &builtin_count](const opt::Instruction* user) {
        if (user->opcode() == spv::Op::OpMemberDecorate &&
            user->GetSingleWordInOperand(0) == struct_type_id &&
            static_cast<spv::Decoration>(user->GetSingleWordInOperand(2)) ==
                spv::Decoration::BuiltIn) {
          ++builtin_count;
        }
      });

  assert((builtin_count == 0 || builtin_count == type_inst->NumInOperands()) &&
         "The module is invalid: either none or all of the members of "
         "|struct_type_id| may be builtin");

  return builtin_count != 0;
}

bool HasBlockOrBufferBlockDecoration(opt::IRContext* ir_context, uint32_t id) {
  for (auto decoration :
       {spv::Decoration::Block, spv::Decoration::BufferBlock}) {
    if (!ir_context->get_decoration_mgr()->WhileEachDecoration(
            id, uint32_t(decoration),
            [](const opt::Instruction & /*unused*/) -> bool {
              return false;
            })) {
      return true;
    }
  }
  return false;
}

bool SplittingBeforeInstructionSeparatesOpSampledImageDefinitionFromUse(
    opt::BasicBlock* block_to_split, opt::Instruction* split_before) {
  std::set<uint32_t> sampled_image_result_ids;
  bool before_split = true;

  // Check all the instructions in the block to split.
  for (auto& instruction : *block_to_split) {
    if (&instruction == &*split_before) {
      before_split = false;
    }
    if (before_split) {
      // If the instruction comes before the split and its opcode is
      // OpSampledImage, record its result id.
      if (instruction.opcode() == spv::Op::OpSampledImage) {
        sampled_image_result_ids.insert(instruction.result_id());
      }
    } else {
      // If the instruction comes after the split, check if ids
      // corresponding to OpSampledImage instructions defined before the split
      // are used, and return true if they are.
      if (!instruction.WhileEachInId(
              [&sampled_image_result_ids](uint32_t* id) -> bool {
                return !sampled_image_result_ids.count(*id);
              })) {
        return true;
      }
    }
  }

  // No usage that would be separated from the definition has been found.
  return false;
}

bool InstructionHasNoSideEffects(const opt::Instruction& instruction) {
  switch (instruction.opcode()) {
    case spv::Op::OpUndef:
    case spv::Op::OpAccessChain:
    case spv::Op::OpInBoundsAccessChain:
    case spv::Op::OpArrayLength:
    case spv::Op::OpVectorExtractDynamic:
    case spv::Op::OpVectorInsertDynamic:
    case spv::Op::OpVectorShuffle:
    case spv::Op::OpCompositeConstruct:
    case spv::Op::OpCompositeExtract:
    case spv::Op::OpCompositeInsert:
    case spv::Op::OpCopyObject:
    case spv::Op::OpTranspose:
    case spv::Op::OpConvertFToU:
    case spv::Op::OpConvertFToS:
    case spv::Op::OpConvertSToF:
    case spv::Op::OpConvertUToF:
    case spv::Op::OpUConvert:
    case spv::Op::OpSConvert:
    case spv::Op::OpFConvert:
    case spv::Op::OpQuantizeToF16:
    case spv::Op::OpSatConvertSToU:
    case spv::Op::OpSatConvertUToS:
    case spv::Op::OpBitcast:
    case spv::Op::OpSNegate:
    case spv::Op::OpFNegate:
    case spv::Op::OpIAdd:
    case spv::Op::OpFAdd:
    case spv::Op::OpISub:
    case spv::Op::OpFSub:
    case spv::Op::OpIMul:
    case spv::Op::OpFMul:
    case spv::Op::OpUDiv:
    case spv::Op::OpSDiv:
    case spv::Op::OpFDiv:
    case spv::Op::OpUMod:
    case spv::Op::OpSRem:
    case spv::Op::OpSMod:
    case spv::Op::OpFRem:
    case spv::Op::OpFMod:
    case spv::Op::OpVectorTimesScalar:
    case spv::Op::OpMatrixTimesScalar:
    case spv::Op::OpVectorTimesMatrix:
    case spv::Op::OpMatrixTimesVector:
    case spv::Op::OpMatrixTimesMatrix:
    case spv::Op::OpOuterProduct:
    case spv::Op::OpDot:
    case spv::Op::OpIAddCarry:
    case spv::Op::OpISubBorrow:
    case spv::Op::OpUMulExtended:
    case spv::Op::OpSMulExtended:
    case spv::Op::OpAny:
    case spv::Op::OpAll:
    case spv::Op::OpIsNan:
    case spv::Op::OpIsInf:
    case spv::Op::OpIsFinite:
    case spv::Op::OpIsNormal:
    case spv::Op::OpSignBitSet:
    case spv::Op::OpLessOrGreater:
    case spv::Op::OpOrdered:
    case spv::Op::OpUnordered:
    case spv::Op::OpLogicalEqual:
    case spv::Op::OpLogicalNotEqual:
    case spv::Op::OpLogicalOr:
    case spv::Op::OpLogicalAnd:
    case spv::Op::OpLogicalNot:
    case spv::Op::OpSelect:
    case spv::Op::OpIEqual:
    case spv::Op::OpINotEqual:
    case spv::Op::OpUGreaterThan:
    case spv::Op::OpSGreaterThan:
    case spv::Op::OpUGreaterThanEqual:
    case spv::Op::OpSGreaterThanEqual:
    case spv::Op::OpULessThan:
    case spv::Op::OpSLessThan:
    case spv::Op::OpULessThanEqual:
    case spv::Op::OpSLessThanEqual:
    case spv::Op::OpFOrdEqual:
    case spv::Op::OpFUnordEqual:
    case spv::Op::OpFOrdNotEqual:
    case spv::Op::OpFUnordNotEqual:
    case spv::Op::OpFOrdLessThan:
    case spv::Op::OpFUnordLessThan:
    case spv::Op::OpFOrdGreaterThan:
    case spv::Op::OpFUnordGreaterThan:
    case spv::Op::OpFOrdLessThanEqual:
    case spv::Op::OpFUnordLessThanEqual:
    case spv::Op::OpFOrdGreaterThanEqual:
    case spv::Op::OpFUnordGreaterThanEqual:
    case spv::Op::OpShiftRightLogical:
    case spv::Op::OpShiftRightArithmetic:
    case spv::Op::OpShiftLeftLogical:
    case spv::Op::OpBitwiseOr:
    case spv::Op::OpBitwiseXor:
    case spv::Op::OpBitwiseAnd:
    case spv::Op::OpNot:
    case spv::Op::OpBitFieldInsert:
    case spv::Op::OpBitFieldSExtract:
    case spv::Op::OpBitFieldUExtract:
    case spv::Op::OpBitReverse:
    case spv::Op::OpBitCount:
    case spv::Op::OpCopyLogical:
    case spv::Op::OpPhi:
    case spv::Op::OpPtrEqual:
    case spv::Op::OpPtrNotEqual:
      return true;
    default:
      return false;
  }
}

std::set<uint32_t> GetReachableReturnBlocks(opt::IRContext* ir_context,
                                            uint32_t function_id) {
  auto function = ir_context->GetFunction(function_id);
  assert(function && "The function |function_id| must exist.");

  std::set<uint32_t> result;

  ir_context->cfg()->ForEachBlockInPostOrder(function->entry().get(),
                                             [&result](opt::BasicBlock* block) {
                                               if (block->IsReturn()) {
                                                 result.emplace(block->id());
                                               }
                                             });

  return result;
}

bool NewTerminatorPreservesDominationRules(opt::IRContext* ir_context,
                                           uint32_t block_id,
                                           opt::Instruction new_terminator) {
  auto* mutated_block = MaybeFindBlock(ir_context, block_id);
  assert(mutated_block && "|block_id| is invalid");

  ChangeTerminatorRAII change_terminator_raii(mutated_block,
                                              std::move(new_terminator));
  opt::DominatorAnalysis dominator_analysis;
  dominator_analysis.InitializeTree(*ir_context->cfg(),
                                    mutated_block->GetParent());

  // Check that each dominator appears before each dominated block.
  std::unordered_map<uint32_t, size_t> positions;
  for (const auto& block : *mutated_block->GetParent()) {
    positions[block.id()] = positions.size();
  }

  std::queue<uint32_t> q({mutated_block->GetParent()->begin()->id()});
  std::unordered_set<uint32_t> visited;
  while (!q.empty()) {
    auto block = q.front();
    q.pop();
    visited.insert(block);

    auto success = ir_context->cfg()->block(block)->WhileEachSuccessorLabel(
        [&positions, &visited, &dominator_analysis, block, &q](uint32_t id) {
          if (id == block) {
            // Handle the case when loop header and continue target are the same
            // block.
            return true;
          }

          if (dominator_analysis.Dominates(block, id) &&
              positions[block] > positions[id]) {
            // |block| dominates |id| but appears after |id| - violates
            // domination rules.
            return false;
          }

          if (!visited.count(id)) {
            q.push(id);
          }

          return true;
        });

    if (!success) {
      return false;
    }
  }

  // For each instruction in the |block->GetParent()| function check whether
  // all its dependencies satisfy domination rules (i.e. all id operands
  // dominate that instruction).
  for (const auto& block : *mutated_block->GetParent()) {
    if (!ir_context->IsReachable(block)) {
      // If some block is not reachable then we don't need to worry about the
      // preservation of domination rules for its instructions.
      continue;
    }

    for (const auto& inst : block) {
      for (uint32_t i = 0; i < inst.NumInOperands();
           i += inst.opcode() == spv::Op::OpPhi ? 2 : 1) {
        const auto& operand = inst.GetInOperand(i);
        if (!spvIsInIdType(operand.type)) {
          continue;
        }

        if (MaybeFindBlock(ir_context, operand.words[0])) {
          // Ignore operands that refer to OpLabel instructions.
          continue;
        }

        const auto* dependency_block =
            ir_context->get_instr_block(operand.words[0]);
        if (!dependency_block) {
          // A global instruction always dominates all instructions in any
          // function.
          continue;
        }

        auto domination_target_id = inst.opcode() == spv::Op::OpPhi
                                        ? inst.GetSingleWordInOperand(i + 1)
                                        : block.id();

        if (!dominator_analysis.Dominates(dependency_block->id(),
                                          domination_target_id)) {
          return false;
        }
      }
    }
  }

  return true;
}

opt::Module::iterator GetFunctionIterator(opt::IRContext* ir_context,
                                          uint32_t function_id) {
  return std::find_if(ir_context->module()->begin(),
                      ir_context->module()->end(),
                      [function_id](const opt::Function& f) {
                        return f.result_id() == function_id;
                      });
}

// TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3582): Add all
//  opcodes that are agnostic to signedness of operands to function.
//  This is not exhaustive yet.
bool IsAgnosticToSignednessOfOperand(spv::Op opcode,
                                     uint32_t use_in_operand_index) {
  switch (opcode) {
    case spv::Op::OpSNegate:
    case spv::Op::OpNot:
    case spv::Op::OpIAdd:
    case spv::Op::OpISub:
    case spv::Op::OpIMul:
    case spv::Op::OpSDiv:
    case spv::Op::OpSRem:
    case spv::Op::OpSMod:
    case spv::Op::OpShiftRightLogical:
    case spv::Op::OpShiftRightArithmetic:
    case spv::Op::OpShiftLeftLogical:
    case spv::Op::OpBitwiseOr:
    case spv::Op::OpBitwiseXor:
    case spv::Op::OpBitwiseAnd:
    case spv::Op::OpIEqual:
    case spv::Op::OpINotEqual:
    case spv::Op::OpULessThan:
    case spv::Op::OpSLessThan:
    case spv::Op::OpUGreaterThan:
    case spv::Op::OpSGreaterThan:
    case spv::Op::OpULessThanEqual:
    case spv::Op::OpSLessThanEqual:
    case spv::Op::OpUGreaterThanEqual:
    case spv::Op::OpSGreaterThanEqual:
      return true;

    case spv::Op::OpAtomicStore:
    case spv::Op::OpAtomicExchange:
    case spv::Op::OpAtomicIAdd:
    case spv::Op::OpAtomicISub:
    case spv::Op::OpAtomicSMin:
    case spv::Op::OpAtomicUMin:
    case spv::Op::OpAtomicSMax:
    case spv::Op::OpAtomicUMax:
    case spv::Op::OpAtomicAnd:
    case spv::Op::OpAtomicOr:
    case spv::Op::OpAtomicXor:
    case spv::Op::OpAtomicFAddEXT:  // Capability AtomicFloat32AddEXT,
      // AtomicFloat64AddEXT.
      assert(use_in_operand_index != 0 &&
             "Signedness check should not occur on a pointer operand.");
      return use_in_operand_index == 1 || use_in_operand_index == 2;

    case spv::Op::OpAtomicCompareExchange:
    case spv::Op::OpAtomicCompareExchangeWeak:  // Capability Kernel.
      assert(use_in_operand_index != 0 &&
             "Signedness check should not occur on a pointer operand.");
      return use_in_operand_index >= 1 && use_in_operand_index <= 3;

    case spv::Op::OpAtomicLoad:
    case spv::Op::OpAtomicIIncrement:
    case spv::Op::OpAtomicIDecrement:
    case spv::Op::OpAtomicFlagTestAndSet:  // Capability Kernel.
    case spv::Op::OpAtomicFlagClear:       // Capability Kernel.
      assert(use_in_operand_index != 0 &&
             "Signedness check should not occur on a pointer operand.");
      return use_in_operand_index >= 1;

    case spv::Op::OpAccessChain:
      // The signedness of indices does not matter.
      return use_in_operand_index > 0;

    default:
      // Conservatively assume that the id cannot be swapped in other
      // instructions.
      return false;
  }
}

bool TypesAreCompatible(opt::IRContext* ir_context, spv::Op opcode,
                        uint32_t use_in_operand_index, uint32_t type_id_1,
                        uint32_t type_id_2) {
  assert(ir_context->get_type_mgr()->GetType(type_id_1) &&
         ir_context->get_type_mgr()->GetType(type_id_2) &&
         "Type ids are invalid");

  return type_id_1 == type_id_2 ||
         (IsAgnosticToSignednessOfOperand(opcode, use_in_operand_index) &&
          fuzzerutil::TypesAreEqualUpToSign(ir_context, type_id_1, type_id_2));
}

}  // namespace fuzzerutil
}  // namespace fuzz
}  // namespace spvtools
