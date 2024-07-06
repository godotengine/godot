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

#include "source/fuzz/fuzzer_pass.h"

#include <set>

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/id_use_descriptor.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_add_constant_boolean.h"
#include "source/fuzz/transformation_add_constant_composite.h"
#include "source/fuzz/transformation_add_constant_null.h"
#include "source/fuzz/transformation_add_constant_scalar.h"
#include "source/fuzz/transformation_add_global_undef.h"
#include "source/fuzz/transformation_add_global_variable.h"
#include "source/fuzz/transformation_add_local_variable.h"
#include "source/fuzz/transformation_add_loop_preheader.h"
#include "source/fuzz/transformation_add_type_boolean.h"
#include "source/fuzz/transformation_add_type_float.h"
#include "source/fuzz/transformation_add_type_function.h"
#include "source/fuzz/transformation_add_type_int.h"
#include "source/fuzz/transformation_add_type_matrix.h"
#include "source/fuzz/transformation_add_type_pointer.h"
#include "source/fuzz/transformation_add_type_struct.h"
#include "source/fuzz/transformation_add_type_vector.h"
#include "source/fuzz/transformation_split_block.h"

namespace spvtools {
namespace fuzz {

FuzzerPass::FuzzerPass(opt::IRContext* ir_context,
                       TransformationContext* transformation_context,
                       FuzzerContext* fuzzer_context,
                       protobufs::TransformationSequence* transformations,
                       bool ignore_inapplicable_transformations)
    : ir_context_(ir_context),
      transformation_context_(transformation_context),
      fuzzer_context_(fuzzer_context),
      transformations_(transformations),
      ignore_inapplicable_transformations_(
          ignore_inapplicable_transformations) {}

FuzzerPass::~FuzzerPass() = default;

std::vector<opt::Instruction*> FuzzerPass::FindAvailableInstructions(
    opt::Function* function, opt::BasicBlock* block,
    const opt::BasicBlock::iterator& inst_it,
    std::function<bool(opt::IRContext*, opt::Instruction*)>
        instruction_is_relevant) const {
  // TODO(afd) The following is (relatively) simple, but may end up being
  //  prohibitively inefficient, as it walks the whole dominator tree for
  //  every instruction that is considered.

  std::vector<opt::Instruction*> result;
  // Consider all global declarations
  for (auto& global : GetIRContext()->module()->types_values()) {
    if (instruction_is_relevant(GetIRContext(), &global)) {
      result.push_back(&global);
    }
  }

  // Consider all function parameters
  function->ForEachParam(
      [this, &instruction_is_relevant, &result](opt::Instruction* param) {
        if (instruction_is_relevant(GetIRContext(), param)) {
          result.push_back(param);
        }
      });

  // Consider all previous instructions in this block
  for (auto prev_inst_it = block->begin(); prev_inst_it != inst_it;
       ++prev_inst_it) {
    if (instruction_is_relevant(GetIRContext(), &*prev_inst_it)) {
      result.push_back(&*prev_inst_it);
    }
  }

  // Walk the dominator tree to consider all instructions from dominating
  // blocks
  auto dominator_analysis = GetIRContext()->GetDominatorAnalysis(function);
  for (auto next_dominator = dominator_analysis->ImmediateDominator(block);
       next_dominator != nullptr;
       next_dominator =
           dominator_analysis->ImmediateDominator(next_dominator)) {
    for (auto& dominating_inst : *next_dominator) {
      if (instruction_is_relevant(GetIRContext(), &dominating_inst)) {
        result.push_back(&dominating_inst);
      }
    }
  }
  return result;
}

void FuzzerPass::ForEachInstructionWithInstructionDescriptor(
    opt::Function* function,
    std::function<
        void(opt::BasicBlock* block, opt::BasicBlock::iterator inst_it,
             const protobufs::InstructionDescriptor& instruction_descriptor)>
        action) {
  // Consider only reachable blocks. We do this in a separate loop to avoid
  // recomputing the dominator analysis every time |action| changes the
  // module.
  std::vector<opt::BasicBlock*> reachable_blocks;

  for (auto& block : *function) {
    if (GetIRContext()->IsReachable(block)) {
      reachable_blocks.push_back(&block);
    }
  }

  for (auto* block : reachable_blocks) {
    // We now consider every instruction in the block, randomly deciding
    // whether to apply a transformation before it.

    // In order for transformations to insert new instructions, they need to
    // be able to identify the instruction to insert before.  We describe an
    // instruction via its opcode, 'opc', a base instruction 'base' that has a
    // result id, and the number of instructions with opcode 'opc' that we
    // should skip when searching from 'base' for the desired instruction.
    // (An instruction that has a result id is represented by its own opcode,
    // itself as 'base', and a skip-count of 0.)
    std::vector<std::tuple<uint32_t, spv::Op, uint32_t>>
        base_opcode_skip_triples;

    // The initial base instruction is the block label.
    uint32_t base = block->id();

    // Counts the number of times we have seen each opcode since we reset the
    // base instruction.
    std::map<spv::Op, uint32_t> skip_count;

    // Consider every instruction in the block.  The label is excluded: it is
    // only necessary to consider it as a base in case the first instruction
    // in the block does not have a result id.
    for (auto inst_it = block->begin(); inst_it != block->end(); ++inst_it) {
      if (inst_it->HasResultId()) {
        // In the case that the instruction has a result id, we use the
        // instruction as its own base, and clear the skip counts we have
        // collected.
        base = inst_it->result_id();
        skip_count.clear();
      }
      const spv::Op opcode = inst_it->opcode();

      // Invoke the provided function, which might apply a transformation.
      action(block, inst_it,
             MakeInstructionDescriptor(
                 base, opcode,
                 skip_count.count(opcode) ? skip_count.at(opcode) : 0));

      if (!inst_it->HasResultId()) {
        skip_count[opcode] =
            skip_count.count(opcode) ? skip_count.at(opcode) + 1 : 1;
      }
    }
  }
}

void FuzzerPass::ForEachInstructionWithInstructionDescriptor(
    std::function<
        void(opt::Function* function, opt::BasicBlock* block,
             opt::BasicBlock::iterator inst_it,
             const protobufs::InstructionDescriptor& instruction_descriptor)>
        action) {
  // Consider every block in every function.
  for (auto& function : *GetIRContext()->module()) {
    ForEachInstructionWithInstructionDescriptor(
        &function,
        [&action, &function](
            opt::BasicBlock* block, opt::BasicBlock::iterator inst_it,
            const protobufs::InstructionDescriptor& instruction_descriptor) {
          action(&function, block, inst_it, instruction_descriptor);
        });
  }
}

void FuzzerPass::ApplyTransformation(const Transformation& transformation) {
  if (ignore_inapplicable_transformations_) {
    // If an applicable-by-construction transformation turns out to be
    // inapplicable, this is a bug in the fuzzer. However, when deploying the
    // fuzzer at scale for finding bugs in SPIR-V processing tools it is
    // desirable to silently ignore such bugs. This code path caters for that
    // scenario.
    if (!transformation.IsApplicable(GetIRContext(),
                                     *GetTransformationContext())) {
      return;
    }
  } else {
    // This code path caters for debugging bugs in the fuzzer, where an
    // applicable-by-construction transformation turns out to be inapplicable.
    assert(transformation.IsApplicable(GetIRContext(),
                                       *GetTransformationContext()) &&
           "Transformation should be applicable by construction.");
  }
  transformation.Apply(GetIRContext(), GetTransformationContext());
  auto transformation_message = transformation.ToMessage();
  assert(transformation_message.transformation_case() !=
             protobufs::Transformation::TRANSFORMATION_NOT_SET &&
         "Bad transformation.");
  *GetTransformations()->add_transformation() =
      std::move(transformation_message);
}

bool FuzzerPass::MaybeApplyTransformation(
    const Transformation& transformation) {
  if (transformation.IsApplicable(GetIRContext(),
                                  *GetTransformationContext())) {
    transformation.Apply(GetIRContext(), GetTransformationContext());
    auto transformation_message = transformation.ToMessage();
    assert(transformation_message.transformation_case() !=
               protobufs::Transformation::TRANSFORMATION_NOT_SET &&
           "Bad transformation.");
    *GetTransformations()->add_transformation() =
        std::move(transformation_message);
    return true;
  }
  return false;
}

uint32_t FuzzerPass::FindOrCreateBoolType() {
  if (auto existing_id = fuzzerutil::MaybeGetBoolType(GetIRContext())) {
    return existing_id;
  }
  auto result = GetFuzzerContext()->GetFreshId();
  ApplyTransformation(TransformationAddTypeBoolean(result));
  return result;
}

uint32_t FuzzerPass::FindOrCreateIntegerType(uint32_t width, bool is_signed) {
  opt::analysis::Integer int_type(width, is_signed);
  auto existing_id = GetIRContext()->get_type_mgr()->GetId(&int_type);
  if (existing_id) {
    return existing_id;
  }
  auto result = GetFuzzerContext()->GetFreshId();
  ApplyTransformation(TransformationAddTypeInt(result, width, is_signed));
  return result;
}

uint32_t FuzzerPass::FindOrCreateFloatType(uint32_t width) {
  opt::analysis::Float float_type(width);
  auto existing_id = GetIRContext()->get_type_mgr()->GetId(&float_type);
  if (existing_id) {
    return existing_id;
  }
  auto result = GetFuzzerContext()->GetFreshId();
  ApplyTransformation(TransformationAddTypeFloat(result, width));
  return result;
}

uint32_t FuzzerPass::FindOrCreateFunctionType(
    uint32_t return_type_id, const std::vector<uint32_t>& argument_id) {
  // FindFunctionType has a single argument for OpTypeFunction operands
  // so we will have to copy them all in this vector
  std::vector<uint32_t> type_ids(argument_id.size() + 1);
  type_ids[0] = return_type_id;
  std::copy(argument_id.begin(), argument_id.end(), type_ids.begin() + 1);

  // Check if type exists
  auto existing_id = fuzzerutil::FindFunctionType(GetIRContext(), type_ids);
  if (existing_id) {
    return existing_id;
  }

  auto result = GetFuzzerContext()->GetFreshId();
  ApplyTransformation(
      TransformationAddTypeFunction(result, return_type_id, argument_id));
  return result;
}

uint32_t FuzzerPass::FindOrCreateVectorType(uint32_t component_type_id,
                                            uint32_t component_count) {
  assert(component_count >= 2 && component_count <= 4 &&
         "Precondition: component count must be in range [2, 4].");
  opt::analysis::Type* component_type =
      GetIRContext()->get_type_mgr()->GetType(component_type_id);
  assert(component_type && "Precondition: the component type must exist.");
  opt::analysis::Vector vector_type(component_type, component_count);
  auto existing_id = GetIRContext()->get_type_mgr()->GetId(&vector_type);
  if (existing_id) {
    return existing_id;
  }
  auto result = GetFuzzerContext()->GetFreshId();
  ApplyTransformation(
      TransformationAddTypeVector(result, component_type_id, component_count));
  return result;
}

uint32_t FuzzerPass::FindOrCreateMatrixType(uint32_t column_count,
                                            uint32_t row_count) {
  assert(column_count >= 2 && column_count <= 4 &&
         "Precondition: column count must be in range [2, 4].");
  assert(row_count >= 2 && row_count <= 4 &&
         "Precondition: row count must be in range [2, 4].");
  uint32_t column_type_id =
      FindOrCreateVectorType(FindOrCreateFloatType(32), row_count);
  opt::analysis::Type* column_type =
      GetIRContext()->get_type_mgr()->GetType(column_type_id);
  opt::analysis::Matrix matrix_type(column_type, column_count);
  auto existing_id = GetIRContext()->get_type_mgr()->GetId(&matrix_type);
  if (existing_id) {
    return existing_id;
  }
  auto result = GetFuzzerContext()->GetFreshId();
  ApplyTransformation(
      TransformationAddTypeMatrix(result, column_type_id, column_count));
  return result;
}

uint32_t FuzzerPass::FindOrCreateStructType(
    const std::vector<uint32_t>& component_type_ids) {
  if (auto existing_id =
          fuzzerutil::MaybeGetStructType(GetIRContext(), component_type_ids)) {
    return existing_id;
  }
  auto new_id = GetFuzzerContext()->GetFreshId();
  ApplyTransformation(TransformationAddTypeStruct(new_id, component_type_ids));
  return new_id;
}

uint32_t FuzzerPass::FindOrCreatePointerType(uint32_t base_type_id,
                                             spv::StorageClass storage_class) {
  // We do not use the type manager here, due to problems related to isomorphic
  // but distinct structs not being regarded as different.
  auto existing_id = fuzzerutil::MaybeGetPointerType(
      GetIRContext(), base_type_id, storage_class);
  if (existing_id) {
    return existing_id;
  }
  auto result = GetFuzzerContext()->GetFreshId();
  ApplyTransformation(
      TransformationAddTypePointer(result, storage_class, base_type_id));
  return result;
}

uint32_t FuzzerPass::FindOrCreatePointerToIntegerType(
    uint32_t width, bool is_signed, spv::StorageClass storage_class) {
  return FindOrCreatePointerType(FindOrCreateIntegerType(width, is_signed),
                                 storage_class);
}

uint32_t FuzzerPass::FindOrCreateIntegerConstant(
    const std::vector<uint32_t>& words, uint32_t width, bool is_signed,
    bool is_irrelevant) {
  auto int_type_id = FindOrCreateIntegerType(width, is_signed);
  if (auto constant_id = fuzzerutil::MaybeGetScalarConstant(
          GetIRContext(), *GetTransformationContext(), words, int_type_id,
          is_irrelevant)) {
    return constant_id;
  }
  auto result = GetFuzzerContext()->GetFreshId();
  ApplyTransformation(TransformationAddConstantScalar(result, int_type_id,
                                                      words, is_irrelevant));
  return result;
}

uint32_t FuzzerPass::FindOrCreateFloatConstant(
    const std::vector<uint32_t>& words, uint32_t width, bool is_irrelevant) {
  auto float_type_id = FindOrCreateFloatType(width);
  if (auto constant_id = fuzzerutil::MaybeGetScalarConstant(
          GetIRContext(), *GetTransformationContext(), words, float_type_id,
          is_irrelevant)) {
    return constant_id;
  }
  auto result = GetFuzzerContext()->GetFreshId();
  ApplyTransformation(TransformationAddConstantScalar(result, float_type_id,
                                                      words, is_irrelevant));
  return result;
}

uint32_t FuzzerPass::FindOrCreateBoolConstant(bool value, bool is_irrelevant) {
  auto bool_type_id = FindOrCreateBoolType();
  if (auto constant_id = fuzzerutil::MaybeGetScalarConstant(
          GetIRContext(), *GetTransformationContext(), {value ? 1u : 0u},
          bool_type_id, is_irrelevant)) {
    return constant_id;
  }
  auto result = GetFuzzerContext()->GetFreshId();
  ApplyTransformation(
      TransformationAddConstantBoolean(result, value, is_irrelevant));
  return result;
}

uint32_t FuzzerPass::FindOrCreateConstant(const std::vector<uint32_t>& words,
                                          uint32_t type_id,
                                          bool is_irrelevant) {
  assert(type_id && "Constant's type id can't be 0.");

  const auto* type = GetIRContext()->get_type_mgr()->GetType(type_id);
  assert(type && "Type does not exist.");

  if (type->AsBool()) {
    assert(words.size() == 1);
    return FindOrCreateBoolConstant(words[0], is_irrelevant);
  } else if (const auto* integer = type->AsInteger()) {
    return FindOrCreateIntegerConstant(words, integer->width(),
                                       integer->IsSigned(), is_irrelevant);
  } else if (const auto* floating = type->AsFloat()) {
    return FindOrCreateFloatConstant(words, floating->width(), is_irrelevant);
  }

  // This assertion will fail in debug build but not in release build
  // so we return 0 to make compiler happy.
  assert(false && "Constant type is not supported");
  return 0;
}

uint32_t FuzzerPass::FindOrCreateCompositeConstant(
    const std::vector<uint32_t>& component_ids, uint32_t type_id,
    bool is_irrelevant) {
  if (auto existing_constant = fuzzerutil::MaybeGetCompositeConstant(
          GetIRContext(), *GetTransformationContext(), component_ids, type_id,
          is_irrelevant)) {
    return existing_constant;
  }
  uint32_t result = GetFuzzerContext()->GetFreshId();
  ApplyTransformation(TransformationAddConstantComposite(
      result, type_id, component_ids, is_irrelevant));
  return result;
}

uint32_t FuzzerPass::FindOrCreateGlobalUndef(uint32_t type_id) {
  for (auto& inst : GetIRContext()->types_values()) {
    if (inst.opcode() == spv::Op::OpUndef && inst.type_id() == type_id) {
      return inst.result_id();
    }
  }
  auto result = GetFuzzerContext()->GetFreshId();
  ApplyTransformation(TransformationAddGlobalUndef(result, type_id));
  return result;
}

uint32_t FuzzerPass::FindOrCreateNullConstant(uint32_t type_id) {
  // Find existing declaration
  opt::analysis::NullConstant null_constant(
      GetIRContext()->get_type_mgr()->GetType(type_id));
  auto existing_constant =
      GetIRContext()->get_constant_mgr()->FindConstant(&null_constant);

  // Return if found
  if (existing_constant) {
    return GetIRContext()
        ->get_constant_mgr()
        ->GetDefiningInstruction(existing_constant)
        ->result_id();
  }

  // Create new if not found
  auto result = GetFuzzerContext()->GetFreshId();
  ApplyTransformation(TransformationAddConstantNull(result, type_id));
  return result;
}

std::pair<std::vector<uint32_t>, std::map<uint32_t, std::vector<uint32_t>>>
FuzzerPass::GetAvailableBasicTypesAndPointers(
    spv::StorageClass storage_class) const {
  // Records all of the basic types available in the module.
  std::set<uint32_t> basic_types;

  // For each basic type, records all the associated pointer types that target
  // the basic type and that have |storage_class| as their storage class.
  std::map<uint32_t, std::vector<uint32_t>> basic_type_to_pointers;

  for (auto& inst : GetIRContext()->types_values()) {
    // For each basic type that we come across, record type, and the fact that
    // we cannot yet have seen any pointers that use the basic type as its
    // pointee type.
    //
    // For pointer types with basic pointee types, associate the pointer type
    // with the basic type.
    switch (inst.opcode()) {
      case spv::Op::OpTypeBool:
      case spv::Op::OpTypeFloat:
      case spv::Op::OpTypeInt:
      case spv::Op::OpTypeMatrix:
      case spv::Op::OpTypeVector:
        // These are all basic types.
        basic_types.insert(inst.result_id());
        basic_type_to_pointers.insert({inst.result_id(), {}});
        break;
      case spv::Op::OpTypeArray:
        // An array type is basic if its base type is basic.
        if (basic_types.count(inst.GetSingleWordInOperand(0))) {
          basic_types.insert(inst.result_id());
          basic_type_to_pointers.insert({inst.result_id(), {}});
        }
        break;
      case spv::Op::OpTypeStruct: {
        // A struct type is basic if it does not have the Block/BufferBlock
        // decoration, and if all of its members are basic.
        if (!fuzzerutil::HasBlockOrBufferBlockDecoration(GetIRContext(),
                                                         inst.result_id())) {
          bool all_members_are_basic_types = true;
          for (uint32_t i = 0; i < inst.NumInOperands(); i++) {
            if (!basic_types.count(inst.GetSingleWordInOperand(i))) {
              all_members_are_basic_types = false;
              break;
            }
          }
          if (all_members_are_basic_types) {
            basic_types.insert(inst.result_id());
            basic_type_to_pointers.insert({inst.result_id(), {}});
          }
        }
        break;
      }
      case spv::Op::OpTypePointer: {
        // We are interested in the pointer if its pointee type is basic and it
        // has the right storage class.
        auto pointee_type = inst.GetSingleWordInOperand(1);
        if (spv::StorageClass(inst.GetSingleWordInOperand(0)) ==
                storage_class &&
            basic_types.count(pointee_type)) {
          // The pointer has the desired storage class, and its pointee type is
          // a basic type, so we are interested in it.  Associate it with its
          // basic type.
          basic_type_to_pointers.at(pointee_type).push_back(inst.result_id());
        }
        break;
      }
      default:
        break;
    }
  }
  return {{basic_types.begin(), basic_types.end()}, basic_type_to_pointers};
}

uint32_t FuzzerPass::FindOrCreateZeroConstant(
    uint32_t scalar_or_composite_type_id, bool is_irrelevant) {
  auto type_instruction =
      GetIRContext()->get_def_use_mgr()->GetDef(scalar_or_composite_type_id);
  assert(type_instruction && "The type instruction must exist.");
  switch (type_instruction->opcode()) {
    case spv::Op::OpTypeBool:
      return FindOrCreateBoolConstant(false, is_irrelevant);
    case spv::Op::OpTypeFloat: {
      auto width = type_instruction->GetSingleWordInOperand(0);
      auto num_words = (width + 32 - 1) / 32;
      return FindOrCreateFloatConstant(std::vector<uint32_t>(num_words, 0),
                                       width, is_irrelevant);
    }
    case spv::Op::OpTypeInt: {
      auto width = type_instruction->GetSingleWordInOperand(0);
      auto num_words = (width + 32 - 1) / 32;
      return FindOrCreateIntegerConstant(
          std::vector<uint32_t>(num_words, 0), width,
          type_instruction->GetSingleWordInOperand(1), is_irrelevant);
    }
    case spv::Op::OpTypeArray: {
      auto component_type_id = type_instruction->GetSingleWordInOperand(0);
      auto num_components =
          fuzzerutil::GetArraySize(*type_instruction, GetIRContext());
      return FindOrCreateCompositeConstant(
          std::vector<uint32_t>(
              num_components,
              FindOrCreateZeroConstant(component_type_id, is_irrelevant)),
          scalar_or_composite_type_id, is_irrelevant);
    }
    case spv::Op::OpTypeMatrix:
    case spv::Op::OpTypeVector: {
      auto component_type_id = type_instruction->GetSingleWordInOperand(0);
      auto num_components = type_instruction->GetSingleWordInOperand(1);
      return FindOrCreateCompositeConstant(
          std::vector<uint32_t>(
              num_components,
              FindOrCreateZeroConstant(component_type_id, is_irrelevant)),
          scalar_or_composite_type_id, is_irrelevant);
    }
    case spv::Op::OpTypeStruct: {
      assert(!fuzzerutil::HasBlockOrBufferBlockDecoration(
                 GetIRContext(), scalar_or_composite_type_id) &&
             "We do not construct constants of struct types decorated with "
             "Block or BufferBlock.");
      std::vector<uint32_t> field_zero_ids;
      for (uint32_t index = 0; index < type_instruction->NumInOperands();
           index++) {
        field_zero_ids.push_back(FindOrCreateZeroConstant(
            type_instruction->GetSingleWordInOperand(index), is_irrelevant));
      }
      return FindOrCreateCompositeConstant(
          field_zero_ids, scalar_or_composite_type_id, is_irrelevant);
    }
    default:
      assert(false && "Unknown type.");
      return 0;
  }
}

void FuzzerPass::MaybeAddUseToReplace(
    opt::Instruction* use_inst, uint32_t use_index, uint32_t replacement_id,
    std::vector<std::pair<protobufs::IdUseDescriptor, uint32_t>>*
        uses_to_replace) {
  // Only consider this use if it is in a block
  if (!GetIRContext()->get_instr_block(use_inst)) {
    return;
  }

  // Get the index of the operand restricted to input operands.
  uint32_t in_operand_index =
      fuzzerutil::InOperandIndexFromOperandIndex(*use_inst, use_index);
  auto id_use_descriptor =
      MakeIdUseDescriptorFromUse(GetIRContext(), use_inst, in_operand_index);
  uses_to_replace->emplace_back(
      std::make_pair(id_use_descriptor, replacement_id));
}

opt::BasicBlock* FuzzerPass::GetOrCreateSimpleLoopPreheader(
    uint32_t header_id) {
  auto header_block = fuzzerutil::MaybeFindBlock(GetIRContext(), header_id);

  assert(header_block && header_block->IsLoopHeader() &&
         "|header_id| should be the label id of a loop header");

  auto predecessors = GetIRContext()->cfg()->preds(header_id);

  assert(predecessors.size() >= 2 &&
         "The block |header_id| should be reachable.");

  auto function = header_block->GetParent();

  if (predecessors.size() == 2) {
    // The header has a single out-of-loop predecessor, which could be a
    // preheader.

    opt::BasicBlock* maybe_preheader;

    if (GetIRContext()->GetDominatorAnalysis(function)->Dominates(
            header_id, predecessors[0])) {
      // The first predecessor is the back-edge block, because the header
      // dominates it, so the second one is out of the loop.
      maybe_preheader = &*function->FindBlock(predecessors[1]);
    } else {
      // The first predecessor is out of the loop.
      maybe_preheader = &*function->FindBlock(predecessors[0]);
    }

    // |maybe_preheader| is a preheader if it branches unconditionally to
    // the header. We also require it not to be a loop header.
    if (maybe_preheader->terminator()->opcode() == spv::Op::OpBranch &&
        !maybe_preheader->IsLoopHeader()) {
      return maybe_preheader;
    }
  }

  // We need to add a preheader.

  // Get a fresh id for the preheader.
  uint32_t preheader_id = GetFuzzerContext()->GetFreshId();

  // Get a fresh id for each OpPhi instruction, if there is more than one
  // out-of-loop predecessor.
  std::vector<uint32_t> phi_ids;
  if (predecessors.size() > 2) {
    header_block->ForEachPhiInst(
        [this, &phi_ids](opt::Instruction* /* unused */) {
          phi_ids.push_back(GetFuzzerContext()->GetFreshId());
        });
  }

  // Add the preheader.
  ApplyTransformation(
      TransformationAddLoopPreheader(header_id, preheader_id, phi_ids));

  // Make the newly-created preheader the new entry block.
  return &*function->FindBlock(preheader_id);
}

opt::BasicBlock* FuzzerPass::SplitBlockAfterOpPhiOrOpVariable(
    uint32_t block_id) {
  auto block = fuzzerutil::MaybeFindBlock(GetIRContext(), block_id);
  assert(block && "|block_id| must be a block label");
  assert(!block->IsLoopHeader() && "|block_id| cannot be a loop header");

  // Find the first non-OpPhi and non-OpVariable instruction.
  auto non_phi_or_var_inst = &*block->begin();
  while (non_phi_or_var_inst->opcode() == spv::Op::OpPhi ||
         non_phi_or_var_inst->opcode() == spv::Op::OpVariable) {
    non_phi_or_var_inst = non_phi_or_var_inst->NextNode();
  }

  // Split the block.
  uint32_t new_block_id = GetFuzzerContext()->GetFreshId();
  ApplyTransformation(TransformationSplitBlock(
      MakeInstructionDescriptor(GetIRContext(), non_phi_or_var_inst),
      new_block_id));

  // We need to return the newly-created block.
  return &*block->GetParent()->FindBlock(new_block_id);
}

uint32_t FuzzerPass::FindOrCreateLocalVariable(
    uint32_t pointer_type_id, uint32_t function_id,
    bool pointee_value_is_irrelevant) {
  auto pointer_type = GetIRContext()->get_type_mgr()->GetType(pointer_type_id);
  // No unused variables in release mode.
  (void)pointer_type;
  assert(pointer_type && pointer_type->AsPointer() &&
         pointer_type->AsPointer()->storage_class() ==
             spv::StorageClass::Function &&
         "The pointer_type_id must refer to a defined pointer type with "
         "storage class Function");
  auto function = fuzzerutil::FindFunction(GetIRContext(), function_id);
  assert(function && "The function must be defined.");

  // First we try to find a suitable existing variable.
  // All of the local variable declarations are located in the first block.
  for (auto& instruction : *function->begin()) {
    if (instruction.opcode() != spv::Op::OpVariable) {
      continue;
    }
    // The existing OpVariable must have type |pointer_type_id|.
    if (instruction.type_id() != pointer_type_id) {
      continue;
    }
    // Check if the found variable is marked with PointeeValueIsIrrelevant
    // according to |pointee_value_is_irrelevant|.
    if (GetTransformationContext()->GetFactManager()->PointeeValueIsIrrelevant(
            instruction.result_id()) != pointee_value_is_irrelevant) {
      continue;
    }
    return instruction.result_id();
  }

  // No such variable was found. Apply a transformation to get one.
  uint32_t pointee_type_id = fuzzerutil::GetPointeeTypeIdFromPointerType(
      GetIRContext(), pointer_type_id);
  uint32_t result_id = GetFuzzerContext()->GetFreshId();
  ApplyTransformation(TransformationAddLocalVariable(
      result_id, pointer_type_id, function_id,
      FindOrCreateZeroConstant(pointee_type_id, pointee_value_is_irrelevant),
      pointee_value_is_irrelevant));
  return result_id;
}

uint32_t FuzzerPass::FindOrCreateGlobalVariable(
    uint32_t pointer_type_id, bool pointee_value_is_irrelevant) {
  auto pointer_type = GetIRContext()->get_type_mgr()->GetType(pointer_type_id);
  // No unused variables in release mode.
  (void)pointer_type;
  assert(
      pointer_type && pointer_type->AsPointer() &&
      (pointer_type->AsPointer()->storage_class() ==
           spv::StorageClass::Private ||
       pointer_type->AsPointer()->storage_class() ==
           spv::StorageClass::Workgroup) &&
      "The pointer_type_id must refer to a defined pointer type with storage "
      "class Private or Workgroup");

  // First we try to find a suitable existing variable.
  for (auto& instruction : GetIRContext()->module()->types_values()) {
    if (instruction.opcode() != spv::Op::OpVariable) {
      continue;
    }
    // The existing OpVariable must have type |pointer_type_id|.
    if (instruction.type_id() != pointer_type_id) {
      continue;
    }
    // Check if the found variable is marked with PointeeValueIsIrrelevant
    // according to |pointee_value_is_irrelevant|.
    if (GetTransformationContext()->GetFactManager()->PointeeValueIsIrrelevant(
            instruction.result_id()) != pointee_value_is_irrelevant) {
      continue;
    }
    return instruction.result_id();
  }

  // No such variable was found. Apply a transformation to get one.
  uint32_t pointee_type_id = fuzzerutil::GetPointeeTypeIdFromPointerType(
      GetIRContext(), pointer_type_id);
  auto storage_class = fuzzerutil::GetStorageClassFromPointerType(
      GetIRContext(), pointer_type_id);
  uint32_t result_id = GetFuzzerContext()->GetFreshId();

  // A variable with storage class Workgroup shouldn't have an initializer.
  if (storage_class == spv::StorageClass::Workgroup) {
    ApplyTransformation(TransformationAddGlobalVariable(
        result_id, pointer_type_id, spv::StorageClass::Workgroup, 0,
        pointee_value_is_irrelevant));
  } else {
    ApplyTransformation(TransformationAddGlobalVariable(
        result_id, pointer_type_id, spv::StorageClass::Private,
        FindOrCreateZeroConstant(pointee_type_id, pointee_value_is_irrelevant),
        pointee_value_is_irrelevant));
  }
  return result_id;
}

}  // namespace fuzz
}  // namespace spvtools
