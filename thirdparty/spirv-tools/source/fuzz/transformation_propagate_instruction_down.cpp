// Copyright (c) 2020 Vasyl Teliman
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

#include "source/fuzz/transformation_propagate_instruction_down.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationPropagateInstructionDown::TransformationPropagateInstructionDown(
    protobufs::TransformationPropagateInstructionDown message)
    : message_(std::move(message)) {}

TransformationPropagateInstructionDown::TransformationPropagateInstructionDown(
    uint32_t block_id, uint32_t phi_fresh_id,
    const std::map<uint32_t, uint32_t>& successor_id_to_fresh_id) {
  message_.set_block_id(block_id);
  message_.set_phi_fresh_id(phi_fresh_id);
  *message_.mutable_successor_id_to_fresh_id() =
      fuzzerutil::MapToRepeatedUInt32Pair(successor_id_to_fresh_id);
}

bool TransformationPropagateInstructionDown::IsApplicable(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context) const {
  // Check that we can apply this transformation to the |block_id|.
  if (!IsApplicableToBlock(ir_context, message_.block_id())) {
    return false;
  }

  const auto successor_id_to_fresh_id =
      fuzzerutil::RepeatedUInt32PairToMap(message_.successor_id_to_fresh_id());

  for (auto id : GetAcceptableSuccessors(ir_context, message_.block_id())) {
    // Each successor must have a fresh id in the |successor_id_to_fresh_id|
    // map, unless overflow ids are available.
    if (!successor_id_to_fresh_id.count(id) &&
        !transformation_context.GetOverflowIdSource()->HasOverflowIds()) {
      return false;
    }
  }

  std::vector<uint32_t> maybe_fresh_ids = {message_.phi_fresh_id()};
  maybe_fresh_ids.reserve(successor_id_to_fresh_id.size());
  for (const auto& entry : successor_id_to_fresh_id) {
    maybe_fresh_ids.push_back(entry.second);
  }

  // All ids must be unique and fresh.
  return !fuzzerutil::HasDuplicates(maybe_fresh_ids) &&
         std::all_of(maybe_fresh_ids.begin(), maybe_fresh_ids.end(),
                     [ir_context](uint32_t id) {
                       return fuzzerutil::IsFreshId(ir_context, id);
                     });
}

void TransformationPropagateInstructionDown::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  // Get instruction to propagate down. There must be one.
  auto* inst_to_propagate =
      GetInstructionToPropagate(ir_context, message_.block_id());
  assert(inst_to_propagate && "There must be an instruction to propagate");

  auto successor_id_to_fresh_id =
      fuzzerutil::RepeatedUInt32PairToMap(message_.successor_id_to_fresh_id());
  std::vector<uint32_t> created_inst_ids;
  auto successor_ids = GetAcceptableSuccessors(ir_context, message_.block_id());

  // Clone |inst_to_propagate| into every successor.
  for (auto successor_id : successor_ids) {
    std::unique_ptr<opt::Instruction> clone(
        inst_to_propagate->Clone(ir_context));

    uint32_t new_result_id;
    if (successor_id_to_fresh_id.count(successor_id)) {
      new_result_id = successor_id_to_fresh_id.at(successor_id);
    } else {
      assert(transformation_context->GetOverflowIdSource()->HasOverflowIds() &&
             "Overflow ids must be available");
      new_result_id =
          transformation_context->GetOverflowIdSource()->GetNextOverflowId();
      successor_id_to_fresh_id[successor_id] = new_result_id;
    }

    clone->SetResultId(new_result_id);
    fuzzerutil::UpdateModuleIdBound(ir_context, new_result_id);

    auto* insert_before_inst = GetFirstInsertBeforeInstruction(
        ir_context, successor_id, clone->opcode());
    assert(insert_before_inst && "Can't insert into one of the successors");

    insert_before_inst->InsertBefore(std::move(clone));
    created_inst_ids.push_back(new_result_id);
  }

  // Add an OpPhi instruction into the module if possible.
  if (auto merge_block_id = GetOpPhiBlockId(
          ir_context, message_.block_id(), *inst_to_propagate, successor_ids)) {
    opt::Instruction::OperandList in_operands;
    std::unordered_set<uint32_t> visited_predecessors;
    for (auto predecessor_id : ir_context->cfg()->preds(merge_block_id)) {
      if (visited_predecessors.count(predecessor_id)) {
        // Merge block might have multiple identical predecessors.
        continue;
      }

      visited_predecessors.insert(predecessor_id);

      const auto* dominator_analysis = ir_context->GetDominatorAnalysis(
          ir_context->cfg()->block(message_.block_id())->GetParent());

      // Find the successor of |source_block| that dominates the predecessor of
      // the merge block |predecessor_id|.
      auto it = std::find_if(
          successor_ids.begin(), successor_ids.end(),
          [predecessor_id, dominator_analysis](uint32_t successor_id) {
            return dominator_analysis->Dominates(successor_id, predecessor_id);
          });

      // OpPhi requires a single operand pair for every predecessor of the
      // OpPhi's block.
      assert(it != successor_ids.end() && "Unable to insert OpPhi");

      in_operands.push_back(
          {SPV_OPERAND_TYPE_ID, {successor_id_to_fresh_id.at(*it)}});
      in_operands.push_back({SPV_OPERAND_TYPE_ID, {predecessor_id}});
    }

    ir_context->cfg()
        ->block(merge_block_id)
        ->begin()
        ->InsertBefore(MakeUnique<opt::Instruction>(
            ir_context, spv::Op::OpPhi, inst_to_propagate->type_id(),
            message_.phi_fresh_id(), std::move(in_operands)));

    fuzzerutil::UpdateModuleIdBound(ir_context, message_.phi_fresh_id());
    created_inst_ids.push_back(message_.phi_fresh_id());
  }

  // Make sure analyses are updated when we adjust users of |inst_to_propagate|.
  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);

  // Copy decorations from the original instructions to its propagated copies.
  for (auto id : created_inst_ids) {
    ir_context->get_decoration_mgr()->CloneDecorations(
        inst_to_propagate->result_id(), id);
  }

  // Remove all decorations from the original instruction.
  ir_context->get_decoration_mgr()->RemoveDecorationsFrom(
      inst_to_propagate->result_id());

  // Update every use of the |inst_to_propagate| with a result id of some of the
  // newly created instructions.
  ir_context->get_def_use_mgr()->ForEachUse(
      inst_to_propagate, [ir_context, &created_inst_ids](
                             opt::Instruction* user, uint32_t operand_index) {
        assert(ir_context->get_instr_block(user) &&
               "All decorations should have already been adjusted");

        auto in_operand_index =
            fuzzerutil::InOperandIndexFromOperandIndex(*user, operand_index);
        for (auto id : created_inst_ids) {
          if (fuzzerutil::IdIsAvailableAtUse(ir_context, user, in_operand_index,
                                             id)) {
            user->SetInOperand(in_operand_index, {id});
            return;
          }
        }

        // Every user of |inst_to_propagate| must be updated since we will
        // remove that instruction from the module.
        assert(false && "Every user of |inst_to_propagate| must be updated");
      });

  // Add synonyms about newly created instructions.
  assert(inst_to_propagate->HasResultId() &&
         "Result id is required to add facts");
  if (transformation_context->GetFactManager()->IdIsIrrelevant(
          inst_to_propagate->result_id())) {
    for (auto id : created_inst_ids) {
      transformation_context->GetFactManager()->AddFactIdIsIrrelevant(id);
    }
  } else {
    std::vector<uint32_t> non_irrelevant_ids;
    for (auto id : created_inst_ids) {
      // |id| can be irrelevant implicitly (e.g. if we propagate it into a dead
      // block).
      if (!transformation_context->GetFactManager()->IdIsIrrelevant(id)) {
        non_irrelevant_ids.push_back(id);
      }
    }

    if (transformation_context->GetFactManager()->PointeeValueIsIrrelevant(
            inst_to_propagate->result_id())) {
      for (auto id : non_irrelevant_ids) {
        transformation_context->GetFactManager()
            ->AddFactValueOfPointeeIsIrrelevant(id);
      }
    }

    for (auto id : non_irrelevant_ids) {
      transformation_context->GetFactManager()->AddFactDataSynonym(
          MakeDataDescriptor(id, {}),
          MakeDataDescriptor(non_irrelevant_ids[0], {}));
    }
  }

  // Remove the propagated instruction from the module.
  ir_context->KillInst(inst_to_propagate);

  // We've adjusted all users - make sure these changes are analyzed.
  ir_context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

protobufs::Transformation TransformationPropagateInstructionDown::ToMessage()
    const {
  protobufs::Transformation result;
  *result.mutable_propagate_instruction_down() = message_;
  return result;
}

bool TransformationPropagateInstructionDown::IsOpcodeSupported(spv::Op opcode) {
  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3605):
  //  We only support "simple" instructions that don't work with memory.
  //  We should extend this so that we support the ones that modify the memory
  //  too.
  switch (opcode) {
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
    case spv::Op::OpPtrEqual:
    case spv::Op::OpPtrNotEqual:
      return true;
    default:
      return false;
  }
}

opt::Instruction*
TransformationPropagateInstructionDown::GetInstructionToPropagate(
    opt::IRContext* ir_context, uint32_t block_id) {
  auto* block = ir_context->cfg()->block(block_id);
  assert(block && "|block_id| is invalid");

  for (auto it = block->rbegin(); it != block->rend(); ++it) {
    if (!it->result_id() || !it->type_id() ||
        !IsOpcodeSupported(it->opcode())) {
      continue;
    }

    auto all_users_from_different_blocks =
        ir_context->get_def_use_mgr()->WhileEachUser(
            &*it, [ir_context, block](opt::Instruction* user) {
              return ir_context->get_instr_block(user) != block;
            });

    if (!all_users_from_different_blocks) {
      // We can't propagate an instruction if it's used in the same block.
      continue;
    }

    return &*it;
  }

  return nullptr;
}

bool TransformationPropagateInstructionDown::IsApplicableToBlock(
    opt::IRContext* ir_context, uint32_t block_id) {
  // Check that |block_id| is valid.
  const auto* block = fuzzerutil::MaybeFindBlock(ir_context, block_id);
  if (!block) {
    return false;
  }

  // |block| must be reachable.
  if (!ir_context->IsReachable(*block)) {
    return false;
  }

  // The block must have an instruction to propagate.
  const auto* inst_to_propagate =
      GetInstructionToPropagate(ir_context, block_id);
  if (!inst_to_propagate) {
    return false;
  }

  // Check that |block| has successors.
  auto successor_ids = GetAcceptableSuccessors(ir_context, block_id);
  if (successor_ids.empty()) {
    return false;
  }

  // Check that |successor_block| doesn't have any OpPhi instructions that
  // use |inst|.
  for (auto successor_id : successor_ids) {
    for (const auto& maybe_phi_inst : *ir_context->cfg()->block(successor_id)) {
      if (maybe_phi_inst.opcode() != spv::Op::OpPhi) {
        // OpPhis can be intermixed with OpLine and OpNoLine.
        continue;
      }

      for (uint32_t i = 0; i < maybe_phi_inst.NumInOperands(); i += 2) {
        if (maybe_phi_inst.GetSingleWordInOperand(i) ==
            inst_to_propagate->result_id()) {
          return false;
        }
      }
    }
  }

  // Get the result id of the block we will insert OpPhi instruction into.
  // This is either 0 or a result id of some merge block in the function.
  auto phi_block_id =
      GetOpPhiBlockId(ir_context, block_id, *inst_to_propagate, successor_ids);

  const auto* dominator_analysis =
      ir_context->GetDominatorAnalysis(block->GetParent());

  // Make sure we can adjust all users of the propagated instruction.
  return ir_context->get_def_use_mgr()->WhileEachUse(
      inst_to_propagate,
      [ir_context, &successor_ids, dominator_analysis, phi_block_id](
          opt::Instruction* user, uint32_t index) {
        const auto* user_block = ir_context->get_instr_block(user);

        if (!user_block) {
          // |user| might be a global instruction (e.g. OpDecorate).
          return true;
        }

        // Check that at least one of the ids in |successor_ids| or a
        // |phi_block_id| dominates |user|'s block (or its predecessor if the
        // user is an OpPhi). We can't use fuzzerutil::IdIsAvailableAtUse since
        // the id in question hasn't yet been created in the module.
        auto block_id_to_dominate = user->opcode() == spv::Op::OpPhi
                                        ? user->GetSingleWordOperand(index + 1)
                                        : user_block->id();

        if (phi_block_id != 0 &&
            dominator_analysis->Dominates(phi_block_id, block_id_to_dominate)) {
          return true;
        }

        return std::any_of(
            successor_ids.begin(), successor_ids.end(),
            [dominator_analysis, block_id_to_dominate](uint32_t id) {
              return dominator_analysis->Dominates(id, block_id_to_dominate);
            });
      });
}

opt::Instruction*
TransformationPropagateInstructionDown::GetFirstInsertBeforeInstruction(
    opt::IRContext* ir_context, uint32_t block_id, spv::Op opcode) {
  auto* block = ir_context->cfg()->block(block_id);

  auto it = block->begin();

  while (it != block->end() &&
         !fuzzerutil::CanInsertOpcodeBeforeInstruction(opcode, it)) {
    ++it;
  }

  return it == block->end() ? nullptr : &*it;
}

std::unordered_set<uint32_t>
TransformationPropagateInstructionDown::GetAcceptableSuccessors(
    opt::IRContext* ir_context, uint32_t block_id) {
  const auto* block = ir_context->cfg()->block(block_id);
  assert(block && "|block_id| is invalid");

  const auto* inst = GetInstructionToPropagate(ir_context, block_id);
  assert(inst && "The block must have an instruction to propagate");

  std::unordered_set<uint32_t> result;
  block->ForEachSuccessorLabel([ir_context, &result,
                                inst](uint32_t successor_id) {
    if (result.count(successor_id)) {
      return;
    }

    auto* successor_block = ir_context->cfg()->block(successor_id);

    // We can't propagate |inst| into |successor_block| if the latter is not
    // dominated by the |inst|'s dependencies.
    if (!inst->WhileEachInId([ir_context, successor_block](const uint32_t* id) {
          return fuzzerutil::IdIsAvailableBeforeInstruction(
              ir_context, &*successor_block->begin(), *id);
        })) {
      return;
    }

    // We don't propagate any "special" instructions (e.g. OpSelectionMerge
    // etc), thus, insertion point must always exist if the module is valid.
    assert(GetFirstInsertBeforeInstruction(ir_context, successor_id,
                                           inst->opcode()) &&
           "There must exist an insertion point.");

    result.insert(successor_id);
  });

  return result;
}

uint32_t TransformationPropagateInstructionDown::GetOpPhiBlockId(
    opt::IRContext* ir_context, uint32_t block_id,
    const opt::Instruction& inst_to_propagate,
    const std::unordered_set<uint32_t>& successor_ids) {
  const auto* block = ir_context->cfg()->block(block_id);

  // |block_id| must belong to some construct.
  auto merge_block_id =
      block->GetMergeInst()
          ? block->GetMergeInst()->GetSingleWordInOperand(0)
          : ir_context->GetStructuredCFGAnalysis()->MergeBlock(block_id);
  if (!merge_block_id) {
    return 0;
  }

  const auto* dominator_analysis =
      ir_context->GetDominatorAnalysis(block->GetParent());

  // Check that |merge_block_id| is reachable in the CFG and |block_id|
  // dominates |merge_block_id|.
  if (!ir_context->IsReachable(*ir_context->cfg()->block(merge_block_id)) ||
      !dominator_analysis->Dominates(block_id, merge_block_id)) {
    return 0;
  }

  // We can't insert an OpPhi into |merge_block_id| if it's an acceptable
  // successor of |block_id|.
  if (successor_ids.count(merge_block_id)) {
    return 0;
  }

  // All predecessors of the merge block must be dominated by at least one
  // successor of the |block_id|.
  assert(!ir_context->cfg()->preds(merge_block_id).empty() &&
         "Merge block must be reachable");
  for (auto predecessor_id : ir_context->cfg()->preds(merge_block_id)) {
    if (std::none_of(
            successor_ids.begin(), successor_ids.end(),
            [dominator_analysis, predecessor_id](uint32_t successor_id) {
              return dominator_analysis->Dominates(successor_id,
                                                   predecessor_id);
            })) {
      return 0;
    }
  }

  const auto* propagate_type =
      ir_context->get_type_mgr()->GetType(inst_to_propagate.type_id());
  assert(propagate_type && "|inst_to_propagate| must have a valid type");

  // VariablePointers capability implicitly declares
  // VariablePointersStorageBuffer. We need those capabilities since otherwise
  // OpPhi instructions cannot have operands of pointer types.
  if (propagate_type->AsPointer() &&
      !ir_context->get_feature_mgr()->HasCapability(
          spv::Capability::VariablePointersStorageBuffer)) {
    return 0;
  }

  return merge_block_id;
}

std::unordered_set<uint32_t>
TransformationPropagateInstructionDown::GetFreshIds() const {
  std::unordered_set<uint32_t> result = {message_.phi_fresh_id()};
  for (const auto& pair : message_.successor_id_to_fresh_id()) {
    result.insert(pair.second());
  }
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
