// Copyright (c) 2017 Google Inc.
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

#include "source/opt/flatten_decoration_pass.h"

#include <cassert>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {

using Words = std::vector<uint32_t>;
using OrderedUsesMap = std::unordered_map<uint32_t, Words>;

Pass::Status FlattenDecorationPass::Process() {
  bool modified = false;

  // The target Id of OpDecorationGroup instructions.
  // We have to track this separately from its uses, in case it
  // has no uses.
  std::unordered_set<uint32_t> group_ids;
  // Maps a decoration group Id to its GroupDecorate targets, in order
  // of appearance.
  OrderedUsesMap normal_uses;
  // Maps a decoration group Id to its GroupMemberDecorate targets and
  // their indices, in of appearance.
  OrderedUsesMap member_uses;

  auto annotations = context()->annotations();

  // On the first pass, record each OpDecorationGroup with its ordered uses.
  // Rely on unordered_map::operator[] to create its entries on first access.
  for (const auto& inst : annotations) {
    switch (inst.opcode()) {
      case spv::Op::OpDecorationGroup:
        group_ids.insert(inst.result_id());
        break;
      case spv::Op::OpGroupDecorate: {
        Words& words = normal_uses[inst.GetSingleWordInOperand(0)];
        for (uint32_t i = 1; i < inst.NumInOperandWords(); i++) {
          words.push_back(inst.GetSingleWordInOperand(i));
        }
      } break;
      case spv::Op::OpGroupMemberDecorate: {
        Words& words = member_uses[inst.GetSingleWordInOperand(0)];
        for (uint32_t i = 1; i < inst.NumInOperandWords(); i++) {
          words.push_back(inst.GetSingleWordInOperand(i));
        }
      } break;
      default:
        break;
    }
  }

  // On the second pass, replace OpDecorationGroup and its uses with
  // equivalent normal and struct member uses.
  auto inst_iter = annotations.begin();
  // We have to re-evaluate the end pointer
  while (inst_iter != context()->annotations().end()) {
    // Should we replace this instruction?
    bool replace = false;
    switch (inst_iter->opcode()) {
      case spv::Op::OpDecorationGroup:
      case spv::Op::OpGroupDecorate:
      case spv::Op::OpGroupMemberDecorate:
        replace = true;
        break;
      case spv::Op::OpDecorate: {
        // If this decoration targets a group, then replace it
        // by sets of normal and member decorations.
        const uint32_t group = inst_iter->GetSingleWordOperand(0);
        const auto normal_uses_iter = normal_uses.find(group);
        if (normal_uses_iter != normal_uses.end()) {
          for (auto target : normal_uses[group]) {
            std::unique_ptr<Instruction> new_inst(inst_iter->Clone(context()));
            new_inst->SetInOperand(0, Words{target});
            inst_iter = inst_iter.InsertBefore(std::move(new_inst));
            ++inst_iter;
            replace = true;
          }
        }
        const auto member_uses_iter = member_uses.find(group);
        if (member_uses_iter != member_uses.end()) {
          const Words& member_id_pairs = (*member_uses_iter).second;
          // The collection is a sequence of pairs.
          assert((member_id_pairs.size() % 2) == 0);
          for (size_t i = 0; i < member_id_pairs.size(); i += 2) {
            // Make an OpMemberDecorate instruction for each (target, member)
            // pair.
            const uint32_t target = member_id_pairs[i];
            const uint32_t member = member_id_pairs[i + 1];
            std::vector<Operand> operands;
            operands.push_back(Operand(SPV_OPERAND_TYPE_ID, {target}));
            operands.push_back(
                Operand(SPV_OPERAND_TYPE_LITERAL_INTEGER, {member}));
            auto decoration_operands_iter = inst_iter->begin();
            decoration_operands_iter++;  // Skip the group target.
            operands.insert(operands.end(), decoration_operands_iter,
                            inst_iter->end());
            std::unique_ptr<Instruction> new_inst(new Instruction(
                context(), spv::Op::OpMemberDecorate, 0, 0, operands));
            inst_iter = inst_iter.InsertBefore(std::move(new_inst));
            ++inst_iter;
            replace = true;
          }
        }
        // If this is an OpDecorate targeting the OpDecorationGroup itself,
        // remove it even if that decoration group itself is not the target of
        // any OpGroupDecorate or OpGroupMemberDecorate.
        if (!replace && group_ids.count(group)) {
          replace = true;
        }
      } break;
      default:
        break;
    }
    if (replace) {
      inst_iter = inst_iter.Erase();
      modified = true;
    } else {
      // Handle the case of decorations unrelated to decoration groups.
      ++inst_iter;
    }
  }

  // Remove OpName instructions which reference the removed group decorations.
  // An OpDecorationGroup instruction might not have been used by an
  // OpGroupDecorate or OpGroupMemberDecorate instruction.
  if (!group_ids.empty()) {
    for (auto debug_inst_iter = context()->debug2_begin();
         debug_inst_iter != context()->debug2_end();) {
      if (debug_inst_iter->opcode() == spv::Op::OpName) {
        const uint32_t target = debug_inst_iter->GetSingleWordOperand(0);
        if (group_ids.count(target)) {
          debug_inst_iter = debug_inst_iter.Erase();
          modified = true;
        } else {
          ++debug_inst_iter;
        }
      }
    }
  }

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
