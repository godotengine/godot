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

#include "source/reduce/remove_struct_member_reduction_opportunity.h"

#include "source/opt/ir_context.h"

namespace spvtools {
namespace reduce {

bool RemoveStructMemberReductionOpportunity::PreconditionHolds() {
  return struct_type_->NumInOperands() == original_number_of_members_;
}

void RemoveStructMemberReductionOpportunity::Apply() {
  std::set<opt::Instruction*> decorations_to_kill;

  // We need to remove decorations that target the removed struct member, and
  // adapt decorations that target later struct members by decrementing the
  // member identifier.  We also need to adapt composite construction
  // instructions so that no id is provided for the member being removed.
  //
  // To do this, we consider every use of the struct type.
  struct_type_->context()->get_def_use_mgr()->ForEachUse(
      struct_type_, [this, &decorations_to_kill](opt::Instruction* user,
                                                 uint32_t /*operand_index*/) {
        switch (user->opcode()) {
          case spv::Op::OpCompositeConstruct:
          case spv::Op::OpConstantComposite:
            // This use is constructing a composite of the struct type, so we
            // must remove the id that was provided for the member we are
            // removing.
            user->RemoveInOperand(member_index_);
            break;
          case spv::Op::OpMemberDecorate:
            // This use is decorating a member of the struct.
            if (user->GetSingleWordInOperand(1) == member_index_) {
              // The member we are removing is being decorated, so we record
              // that we need to get rid of the decoration.
              decorations_to_kill.insert(user);
            } else if (user->GetSingleWordInOperand(1) > member_index_) {
              // A member beyond the one we are removing is being decorated, so
              // we adjust the index that identifies the member.
              user->SetInOperand(1, {user->GetSingleWordInOperand(1) - 1});
            }
            break;
          default:
            break;
        }
      });

  // Get rid of all the decorations that were found to target the member being
  // removed.
  for (auto decoration_to_kill : decorations_to_kill) {
    decoration_to_kill->context()->KillInst(decoration_to_kill);
  }

  // We now look through all instructions that access composites via sequences
  // of indices. Every time we find an index into the struct whose member is
  // being removed, and if the member being accessed comes after the member
  // being removed, we need to adjust the index accordingly.
  //
  // We go through every relevant instruction in every block of every function,
  // and invoke a helper to adjust it.
  auto context = struct_type_->context();
  for (auto& function : *context->module()) {
    for (auto& block : function) {
      for (auto& inst : block) {
        switch (inst.opcode()) {
          case spv::Op::OpAccessChain:
          case spv::Op::OpInBoundsAccessChain: {
            // These access chain instructions take sequences of ids for
            // indexing, starting from input operand 1.
            auto composite_type_id =
                context->get_def_use_mgr()
                    ->GetDef(context->get_def_use_mgr()
                                 ->GetDef(inst.GetSingleWordInOperand(0))
                                 ->type_id())
                    ->GetSingleWordInOperand(1);
            AdjustAccessedIndices(composite_type_id, 1, false, context, &inst);
          } break;
          case spv::Op::OpPtrAccessChain:
          case spv::Op::OpInBoundsPtrAccessChain: {
            // These access chain instructions take sequences of ids for
            // indexing, starting from input operand 2.
            auto composite_type_id =
                context->get_def_use_mgr()
                    ->GetDef(context->get_def_use_mgr()
                                 ->GetDef(inst.GetSingleWordInOperand(1))
                                 ->type_id())
                    ->GetSingleWordInOperand(1);
            AdjustAccessedIndices(composite_type_id, 2, false, context, &inst);
          } break;
          case spv::Op::OpCompositeExtract: {
            // OpCompositeExtract uses literals for indexing, starting at input
            // operand 1.
            auto composite_type_id =
                context->get_def_use_mgr()
                    ->GetDef(inst.GetSingleWordInOperand(0))
                    ->type_id();
            AdjustAccessedIndices(composite_type_id, 1, true, context, &inst);
          } break;
          case spv::Op::OpCompositeInsert: {
            // OpCompositeInsert uses literals for indexing, starting at input
            // operand 2.
            auto composite_type_id =
                context->get_def_use_mgr()
                    ->GetDef(inst.GetSingleWordInOperand(1))
                    ->type_id();
            AdjustAccessedIndices(composite_type_id, 2, true, context, &inst);
          } break;
          default:
            break;
        }
      }
    }
  }

  // Remove the member from the struct type.
  struct_type_->RemoveInOperand(member_index_);

  context->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
}

void RemoveStructMemberReductionOpportunity::AdjustAccessedIndices(
    uint32_t composite_type_id, uint32_t first_index_input_operand,
    bool literal_indices, opt::IRContext* context,
    opt::Instruction* composite_access_instruction) const {
  // Walk the series of types that are encountered by following the
  // instruction's sequence of indices. For all types except structs, this is
  // routine: the type of the composite dictates what the next type will be
  // regardless of the specific index value.
  uint32_t next_type = composite_type_id;
  for (uint32_t i = first_index_input_operand;
       i < composite_access_instruction->NumInOperands(); i++) {
    auto type_inst = context->get_def_use_mgr()->GetDef(next_type);
    switch (type_inst->opcode()) {
      case spv::Op::OpTypeArray:
      case spv::Op::OpTypeMatrix:
      case spv::Op::OpTypeRuntimeArray:
      case spv::Op::OpTypeVector:
        next_type = type_inst->GetSingleWordInOperand(0);
        break;
      case spv::Op::OpTypeStruct: {
        // Struct types are special because (a) we may need to adjust the index
        // being used, if the struct type is the one from which we are removing
        // a member, and (b) the type encountered by following the current index
        // is dependent on the value of the index.

        // Work out the member being accessed.  If literal indexing is used this
        // is simple; otherwise we need to look up the id of the constant
        // instruction being used as an index and get the value of the constant.
        uint32_t index_operand =
            composite_access_instruction->GetSingleWordInOperand(i);
        uint32_t member = literal_indices ? index_operand
                                          : context->get_def_use_mgr()
                                                ->GetDef(index_operand)
                                                ->GetSingleWordInOperand(0);

        // The next type we will consider is obtained by looking up the struct
        // type at |member|.
        next_type = type_inst->GetSingleWordInOperand(member);

        if (type_inst == struct_type_ && member > member_index_) {
          // The struct type is the struct from which we are removing a member,
          // and the member being accessed is beyond the member we are removing.
          // We thus need to decrement the index by 1.
          uint32_t new_in_operand;
          if (literal_indices) {
            // With literal indexing this is straightforward.
            new_in_operand = member - 1;
          } else {
            // With id-based indexing this is more tricky: we need to find or
            // create a constant instruction whose value is one less than
            // |member|, and use the id of this constant as the replacement
            // input operand.
            auto constant_inst =
                context->get_def_use_mgr()->GetDef(index_operand);
            auto int_type = context->get_type_mgr()
                                ->GetType(constant_inst->type_id())
                                ->AsInteger();
            auto new_index_constant =
                opt::analysis::IntConstant(int_type, {member - 1});
            new_in_operand = context->get_constant_mgr()
                                 ->GetDefiningInstruction(&new_index_constant)
                                 ->result_id();
          }
          composite_access_instruction->SetInOperand(i, {new_in_operand});
        }
      } break;
      default:
        assert(0 && "Unknown composite type.");
        break;
    }
  }
}

}  // namespace reduce
}  // namespace spvtools
