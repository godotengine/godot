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

#include "source/reduce/remove_unused_struct_member_reduction_opportunity_finder.h"

#include <map>
#include <set>

#include "source/reduce/remove_struct_member_reduction_opportunity.h"

namespace spvtools {
namespace reduce {

std::vector<std::unique_ptr<ReductionOpportunity>>
RemoveUnusedStructMemberReductionOpportunityFinder::GetAvailableOpportunities(
    opt::IRContext* context, uint32_t target_function) const {
  if (target_function) {
    // Removing an unused struct member is a global change, as struct types are
    // global.  We thus do not consider such opportunities if we are targeting
    // a specific function.
    return {};
  }

  std::vector<std::unique_ptr<ReductionOpportunity>> result;

  // We track those struct members that are never accessed.  We do this by
  // associating a member index to all the structs that have this member index
  // but do not use it.  This representation is designed to allow reduction
  // opportunities to be provided in a useful manner, so that opportunities
  // associated with the same struct are unlikely to be adjacent.
  std::map<uint32_t, std::set<opt::Instruction*>> unused_member_to_structs;

  // Consider every struct type in the module.
  for (auto& type_or_value : context->types_values()) {
    if (type_or_value.opcode() != spv::Op::OpTypeStruct) {
      continue;
    }

    // Initially, we assume that *every* member of the struct is unused.  We
    // then refine this based on observed uses.
    std::set<uint32_t> unused_members;
    for (uint32_t i = 0; i < type_or_value.NumInOperands(); i++) {
      unused_members.insert(i);
    }

    // A separate reduction pass deals with removal of names.  If a struct
    // member is still named, we treat it as being used.
    context->get_def_use_mgr()->ForEachUse(
        &type_or_value,
        [&unused_members](opt::Instruction* user, uint32_t /*operand_index*/) {
          switch (user->opcode()) {
            case spv::Op::OpMemberName:
              unused_members.erase(user->GetSingleWordInOperand(1));
              break;
            default:
              break;
          }
        });

    for (uint32_t member : unused_members) {
      if (!unused_member_to_structs.count(member)) {
        unused_member_to_structs.insert(
            {member, std::set<opt::Instruction*>()});
      }
      unused_member_to_structs.at(member).insert(&type_or_value);
    }
  }

  // We now go through every instruction that might index into a struct, and
  // refine our tracking of which struct members are used based on the struct
  // indexing we observe.  We cannot just go through all uses of a struct type
  // because the type is not necessarily even referenced, e.g. when walking
  // arrays of structs.
  for (auto& function : *context->module()) {
    for (auto& block : function) {
      for (auto& inst : block) {
        switch (inst.opcode()) {
          // For each indexing operation we observe, we invoke a helper to
          // remove from our map those struct indices that are found to be used.
          // The way the helper is invoked depends on whether the instruction
          // uses literal or id indices, and the offset into the instruction's
          // input operands from which index operands are provided.
          case spv::Op::OpAccessChain:
          case spv::Op::OpInBoundsAccessChain: {
            auto composite_type_id =
                context->get_def_use_mgr()
                    ->GetDef(context->get_def_use_mgr()
                                 ->GetDef(inst.GetSingleWordInOperand(0))
                                 ->type_id())
                    ->GetSingleWordInOperand(1);
            MarkAccessedMembersAsUsed(context, composite_type_id, 1, false,
                                      inst, &unused_member_to_structs);
          } break;
          case spv::Op::OpPtrAccessChain:
          case spv::Op::OpInBoundsPtrAccessChain: {
            auto composite_type_id =
                context->get_def_use_mgr()
                    ->GetDef(context->get_def_use_mgr()
                                 ->GetDef(inst.GetSingleWordInOperand(1))
                                 ->type_id())
                    ->GetSingleWordInOperand(1);
            MarkAccessedMembersAsUsed(context, composite_type_id, 2, false,
                                      inst, &unused_member_to_structs);
          } break;
          case spv::Op::OpCompositeExtract: {
            auto composite_type_id =
                context->get_def_use_mgr()
                    ->GetDef(inst.GetSingleWordInOperand(0))
                    ->type_id();
            MarkAccessedMembersAsUsed(context, composite_type_id, 1, true, inst,
                                      &unused_member_to_structs);
          } break;
          case spv::Op::OpCompositeInsert: {
            auto composite_type_id =
                context->get_def_use_mgr()
                    ->GetDef(inst.GetSingleWordInOperand(1))
                    ->type_id();
            MarkAccessedMembersAsUsed(context, composite_type_id, 2, true, inst,
                                      &unused_member_to_structs);
          } break;
          default:
            break;
        }
      }
    }
  }

  // We now know those struct indices that are unused, and we make a reduction
  // opportunity for each of them. By mapping each relevant member index to the
  // structs in which it is unused, we will group all opportunities to remove
  // member k of a struct (for some k) together.  This reduces the likelihood
  // that opportunities to remove members from the same struct will be adjacent,
  // which is good because such opportunities mutually disable one another.
  for (auto& entry : unused_member_to_structs) {
    for (auto struct_type : entry.second) {
      result.push_back(MakeUnique<RemoveStructMemberReductionOpportunity>(
          struct_type, entry.first));
    }
  }
  return result;
}

void RemoveUnusedStructMemberReductionOpportunityFinder::
    MarkAccessedMembersAsUsed(
        opt::IRContext* context, uint32_t composite_type_id,
        uint32_t first_index_in_operand, bool literal_indices,
        const opt::Instruction& composite_access_instruction,
        std::map<uint32_t, std::set<opt::Instruction*>>*
            unused_member_to_structs) const {
  uint32_t next_type = composite_type_id;
  for (uint32_t i = first_index_in_operand;
       i < composite_access_instruction.NumInOperands(); i++) {
    auto type_inst = context->get_def_use_mgr()->GetDef(next_type);
    switch (type_inst->opcode()) {
      case spv::Op::OpTypeArray:
      case spv::Op::OpTypeMatrix:
      case spv::Op::OpTypeRuntimeArray:
      case spv::Op::OpTypeVector:
        next_type = type_inst->GetSingleWordInOperand(0);
        break;
      case spv::Op::OpTypeStruct: {
        uint32_t index_operand =
            composite_access_instruction.GetSingleWordInOperand(i);
        uint32_t member = literal_indices ? index_operand
                                          : context->get_def_use_mgr()
                                                ->GetDef(index_operand)
                                                ->GetSingleWordInOperand(0);
        // Remove the struct type from the struct types associated with this
        // member index, but only if a set of struct types is known to be
        // associated with this member index.
        if (unused_member_to_structs->count(member)) {
          unused_member_to_structs->at(member).erase(type_inst);
        }
        next_type = type_inst->GetSingleWordInOperand(member);
      } break;
      default:
        assert(0 && "Unknown composite type.");
        break;
    }
  }
}

std::string RemoveUnusedStructMemberReductionOpportunityFinder::GetName()
    const {
  return "RemoveUnusedStructMemberReductionOpportunityFinder";
}

}  // namespace reduce
}  // namespace spvtools
