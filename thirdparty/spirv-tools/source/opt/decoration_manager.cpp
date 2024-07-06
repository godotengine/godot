// Copyright (c) 2017 Pierre Moreau
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

#include "source/opt/decoration_manager.h"

#include <algorithm>
#include <memory>
#include <set>
#include <stack>
#include <utility>

#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {
namespace analysis {
namespace {
using InstructionVector = std::vector<const spvtools::opt::Instruction*>;
using DecorationSet = std::set<std::u32string>;

// Returns true if |a| is a subet of |b|.
bool IsSubset(const DecorationSet& a, const DecorationSet& b) {
  auto it1 = a.begin();
  auto it2 = b.begin();

  while (it1 != a.end()) {
    if (it2 == b.end() || *it1 < *it2) {
      // |*it1| is in |a|, but not in |b|.
      return false;
    }
    if (*it1 == *it2) {
      // Found the element move to the next one.
      it1++;
      it2++;
    } else /* *it1 > *it2 */ {
      // Did not find |*it1| yet, check the next element in |b|.
      it2++;
    }
  }
  return true;
}
}  // namespace

bool DecorationManager::RemoveDecorationsFrom(
    uint32_t id, std::function<bool(const Instruction&)> pred) {
  bool was_modified = false;
  const auto ids_iter = id_to_decoration_insts_.find(id);
  if (ids_iter == id_to_decoration_insts_.end()) {
    return was_modified;
  }

  TargetData& decorations_info = ids_iter->second;
  auto context = module_->context();
  std::vector<Instruction*> insts_to_kill;
  const bool is_group = !decorations_info.decorate_insts.empty();

  // Schedule all direct decorations for removal if instructed as such by
  // |pred|.
  for (Instruction* inst : decorations_info.direct_decorations)
    if (pred(*inst)) insts_to_kill.push_back(inst);

  // For all groups being directly applied to |id|, remove |id| (and the
  // literal if |inst| is an OpGroupMemberDecorate) from the instruction
  // applying the group.
  std::unordered_set<const Instruction*> indirect_decorations_to_remove;
  for (Instruction* inst : decorations_info.indirect_decorations) {
    assert(inst->opcode() == spv::Op::OpGroupDecorate ||
           inst->opcode() == spv::Op::OpGroupMemberDecorate);

    std::vector<Instruction*> group_decorations_to_keep;
    const uint32_t group_id = inst->GetSingleWordInOperand(0u);
    const auto group_iter = id_to_decoration_insts_.find(group_id);
    assert(group_iter != id_to_decoration_insts_.end() &&
           "Unknown decoration group");
    const auto& group_decorations = group_iter->second.direct_decorations;
    for (Instruction* decoration : group_decorations) {
      if (!pred(*decoration)) group_decorations_to_keep.push_back(decoration);
    }

    // If all decorations should be kept, then we can keep |id| part of the
    // group.  However, if the group itself has no decorations, we should remove
    // the id from the group.  This is needed to make |KillNameAndDecorate| work
    // correctly when a decoration group has no decorations.
    if (group_decorations_to_keep.size() == group_decorations.size() &&
        group_decorations.size() != 0) {
      continue;
    }

    // Otherwise, remove |id| from the targets of |group_id|
    const uint32_t stride =
        inst->opcode() == spv::Op::OpGroupDecorate ? 1u : 2u;
    for (uint32_t i = 1u; i < inst->NumInOperands();) {
      if (inst->GetSingleWordInOperand(i) != id) {
        i += stride;
        continue;
      }

      const uint32_t last_operand_index = inst->NumInOperands() - stride;
      if (i < last_operand_index)
        inst->GetInOperand(i) = inst->GetInOperand(last_operand_index);
      // Remove the associated literal, if it exists.
      if (stride == 2u) {
        if (i < last_operand_index)
          inst->GetInOperand(i + 1u) =
              inst->GetInOperand(last_operand_index + 1u);
        inst->RemoveInOperand(last_operand_index + 1u);
      }
      inst->RemoveInOperand(last_operand_index);
      was_modified = true;
    }

    // If the instruction has no targets left, remove the instruction
    // altogether.
    if (inst->NumInOperands() == 1u) {
      indirect_decorations_to_remove.emplace(inst);
      insts_to_kill.push_back(inst);
    } else if (was_modified) {
      context->ForgetUses(inst);
      indirect_decorations_to_remove.emplace(inst);
      context->AnalyzeUses(inst);
    }

    // If only some of the decorations should be kept, clone them and apply
    // them directly to |id|.
    if (!group_decorations_to_keep.empty()) {
      for (Instruction* decoration : group_decorations_to_keep) {
        // simply clone decoration and change |group_id| to |id|
        std::unique_ptr<Instruction> new_inst(
            decoration->Clone(module_->context()));
        new_inst->SetInOperand(0, {id});
        module_->AddAnnotationInst(std::move(new_inst));
        auto decoration_iter = --module_->annotation_end();
        context->AnalyzeUses(&*decoration_iter);
      }
    }
  }

  auto& indirect_decorations = decorations_info.indirect_decorations;
  indirect_decorations.erase(
      std::remove_if(
          indirect_decorations.begin(), indirect_decorations.end(),
          [&indirect_decorations_to_remove](const Instruction* inst) {
            return indirect_decorations_to_remove.count(inst);
          }),
      indirect_decorations.end());

  was_modified |= !insts_to_kill.empty();
  for (Instruction* inst : insts_to_kill) context->KillInst(inst);
  insts_to_kill.clear();

  // Schedule all instructions applying the group for removal if this group no
  // longer applies decorations, either directly or indirectly.
  if (is_group && decorations_info.direct_decorations.empty() &&
      decorations_info.indirect_decorations.empty()) {
    for (Instruction* inst : decorations_info.decorate_insts)
      insts_to_kill.push_back(inst);
  }
  was_modified |= !insts_to_kill.empty();
  for (Instruction* inst : insts_to_kill) context->KillInst(inst);

  if (decorations_info.direct_decorations.empty() &&
      decorations_info.indirect_decorations.empty() &&
      decorations_info.decorate_insts.empty()) {
    id_to_decoration_insts_.erase(ids_iter);
  }
  return was_modified;
}

std::vector<Instruction*> DecorationManager::GetDecorationsFor(
    uint32_t id, bool include_linkage) {
  return InternalGetDecorationsFor<Instruction*>(id, include_linkage);
}

std::vector<const Instruction*> DecorationManager::GetDecorationsFor(
    uint32_t id, bool include_linkage) const {
  return const_cast<DecorationManager*>(this)
      ->InternalGetDecorationsFor<const Instruction*>(id, include_linkage);
}

bool DecorationManager::HaveTheSameDecorations(uint32_t id1,
                                               uint32_t id2) const {
  const InstructionVector decorations_for1 = GetDecorationsFor(id1, false);
  const InstructionVector decorations_for2 = GetDecorationsFor(id2, false);

  // This function splits the decoration instructions into different sets,
  // based on their opcode; only OpDecorate, OpDecorateId,
  // OpDecorateStringGOOGLE, and OpMemberDecorate are considered, the other
  // opcodes are ignored.
  const auto fillDecorationSets =
      [](const InstructionVector& decoration_list, DecorationSet* decorate_set,
         DecorationSet* decorate_id_set, DecorationSet* decorate_string_set,
         DecorationSet* member_decorate_set) {
        for (const Instruction* inst : decoration_list) {
          std::u32string decoration_payload;
          // Ignore the opcode and the target as we do not want them to be
          // compared.
          for (uint32_t i = 1u; i < inst->NumInOperands(); ++i) {
            for (uint32_t word : inst->GetInOperand(i).words) {
              decoration_payload.push_back(word);
            }
          }

          switch (inst->opcode()) {
            case spv::Op::OpDecorate:
              decorate_set->emplace(std::move(decoration_payload));
              break;
            case spv::Op::OpMemberDecorate:
              member_decorate_set->emplace(std::move(decoration_payload));
              break;
            case spv::Op::OpDecorateId:
              decorate_id_set->emplace(std::move(decoration_payload));
              break;
            case spv::Op::OpDecorateStringGOOGLE:
              decorate_string_set->emplace(std::move(decoration_payload));
              break;
            default:
              break;
          }
        }
      };

  DecorationSet decorate_set_for1;
  DecorationSet decorate_id_set_for1;
  DecorationSet decorate_string_set_for1;
  DecorationSet member_decorate_set_for1;
  fillDecorationSets(decorations_for1, &decorate_set_for1,
                     &decorate_id_set_for1, &decorate_string_set_for1,
                     &member_decorate_set_for1);

  DecorationSet decorate_set_for2;
  DecorationSet decorate_id_set_for2;
  DecorationSet decorate_string_set_for2;
  DecorationSet member_decorate_set_for2;
  fillDecorationSets(decorations_for2, &decorate_set_for2,
                     &decorate_id_set_for2, &decorate_string_set_for2,
                     &member_decorate_set_for2);

  const bool result = decorate_set_for1 == decorate_set_for2 &&
                      decorate_id_set_for1 == decorate_id_set_for2 &&
                      member_decorate_set_for1 == member_decorate_set_for2 &&
                      // Compare string sets last in case the strings are long.
                      decorate_string_set_for1 == decorate_string_set_for2;
  return result;
}

bool DecorationManager::HaveSubsetOfDecorations(uint32_t id1,
                                                uint32_t id2) const {
  const InstructionVector decorations_for1 = GetDecorationsFor(id1, false);
  const InstructionVector decorations_for2 = GetDecorationsFor(id2, false);

  // This function splits the decoration instructions into different sets,
  // based on their opcode; only OpDecorate, OpDecorateId,
  // OpDecorateStringGOOGLE, and OpMemberDecorate are considered, the other
  // opcodes are ignored.
  const auto fillDecorationSets =
      [](const InstructionVector& decoration_list, DecorationSet* decorate_set,
         DecorationSet* decorate_id_set, DecorationSet* decorate_string_set,
         DecorationSet* member_decorate_set) {
        for (const Instruction* inst : decoration_list) {
          std::u32string decoration_payload;
          // Ignore the opcode and the target as we do not want them to be
          // compared.
          for (uint32_t i = 1u; i < inst->NumInOperands(); ++i) {
            for (uint32_t word : inst->GetInOperand(i).words) {
              decoration_payload.push_back(word);
            }
          }

          switch (inst->opcode()) {
            case spv::Op::OpDecorate:
              decorate_set->emplace(std::move(decoration_payload));
              break;
            case spv::Op::OpMemberDecorate:
              member_decorate_set->emplace(std::move(decoration_payload));
              break;
            case spv::Op::OpDecorateId:
              decorate_id_set->emplace(std::move(decoration_payload));
              break;
            case spv::Op::OpDecorateStringGOOGLE:
              decorate_string_set->emplace(std::move(decoration_payload));
              break;
            default:
              break;
          }
        }
      };

  DecorationSet decorate_set_for1;
  DecorationSet decorate_id_set_for1;
  DecorationSet decorate_string_set_for1;
  DecorationSet member_decorate_set_for1;
  fillDecorationSets(decorations_for1, &decorate_set_for1,
                     &decorate_id_set_for1, &decorate_string_set_for1,
                     &member_decorate_set_for1);

  DecorationSet decorate_set_for2;
  DecorationSet decorate_id_set_for2;
  DecorationSet decorate_string_set_for2;
  DecorationSet member_decorate_set_for2;
  fillDecorationSets(decorations_for2, &decorate_set_for2,
                     &decorate_id_set_for2, &decorate_string_set_for2,
                     &member_decorate_set_for2);

  const bool result =
      IsSubset(decorate_set_for1, decorate_set_for2) &&
      IsSubset(decorate_id_set_for1, decorate_id_set_for2) &&
      IsSubset(member_decorate_set_for1, member_decorate_set_for2) &&
      // Compare string sets last in case the strings are long.
      IsSubset(decorate_string_set_for1, decorate_string_set_for2);
  return result;
}

// TODO(pierremoreau): If OpDecorateId is referencing an OpConstant, one could
//                     check that the constants are the same rather than just
//                     looking at the constant ID.
bool DecorationManager::AreDecorationsTheSame(const Instruction* inst1,
                                              const Instruction* inst2,
                                              bool ignore_target) const {
  switch (inst1->opcode()) {
    case spv::Op::OpDecorate:
    case spv::Op::OpMemberDecorate:
    case spv::Op::OpDecorateId:
    case spv::Op::OpDecorateStringGOOGLE:
      break;
    default:
      return false;
  }

  if (inst1->opcode() != inst2->opcode() ||
      inst1->NumInOperands() != inst2->NumInOperands())
    return false;

  for (uint32_t i = ignore_target ? 1u : 0u; i < inst1->NumInOperands(); ++i)
    if (inst1->GetInOperand(i) != inst2->GetInOperand(i)) return false;

  return true;
}

void DecorationManager::AnalyzeDecorations() {
  if (!module_) return;

  // For each group and instruction, collect all their decoration instructions.
  for (Instruction& inst : module_->annotations()) {
    AddDecoration(&inst);
  }
}

void DecorationManager::AddDecoration(Instruction* inst) {
  switch (inst->opcode()) {
    case spv::Op::OpDecorate:
    case spv::Op::OpDecorateId:
    case spv::Op::OpDecorateStringGOOGLE:
    case spv::Op::OpMemberDecorate: {
      const auto target_id = inst->GetSingleWordInOperand(0u);
      id_to_decoration_insts_[target_id].direct_decorations.push_back(inst);
      break;
    }
    case spv::Op::OpGroupDecorate:
    case spv::Op::OpGroupMemberDecorate: {
      const uint32_t start =
          inst->opcode() == spv::Op::OpGroupDecorate ? 1u : 2u;
      const uint32_t stride = start;
      for (uint32_t i = start; i < inst->NumInOperands(); i += stride) {
        const auto target_id = inst->GetSingleWordInOperand(i);
        TargetData& target_data = id_to_decoration_insts_[target_id];
        target_data.indirect_decorations.push_back(inst);
      }
      const auto target_id = inst->GetSingleWordInOperand(0u);
      id_to_decoration_insts_[target_id].decorate_insts.push_back(inst);
      break;
    }
    default:
      break;
  }
}

void DecorationManager::AddDecoration(spv::Op opcode,
                                      std::vector<Operand> opnds) {
  IRContext* ctx = module_->context();
  std::unique_ptr<Instruction> newDecoOp(
      new Instruction(ctx, opcode, 0, 0, opnds));
  ctx->AddAnnotationInst(std::move(newDecoOp));
}

void DecorationManager::AddDecoration(uint32_t inst_id, uint32_t decoration) {
  AddDecoration(
      spv::Op::OpDecorate,
      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {inst_id}},
       {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER, {decoration}}});
}

void DecorationManager::AddDecorationVal(uint32_t inst_id, uint32_t decoration,
                                         uint32_t decoration_value) {
  AddDecoration(
      spv::Op::OpDecorate,
      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {inst_id}},
       {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER, {decoration}},
       {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
        {decoration_value}}});
}

void DecorationManager::AddMemberDecoration(uint32_t inst_id, uint32_t member,
                                            uint32_t decoration,
                                            uint32_t decoration_value) {
  AddDecoration(
      spv::Op::OpMemberDecorate,
      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {inst_id}},
       {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER, {member}},
       {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER, {decoration}},
       {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
        {decoration_value}}});
}

template <typename T>
std::vector<T> DecorationManager::InternalGetDecorationsFor(
    uint32_t id, bool include_linkage) {
  std::vector<T> decorations;

  const auto ids_iter = id_to_decoration_insts_.find(id);
  // |id| has no decorations
  if (ids_iter == id_to_decoration_insts_.end()) return decorations;

  const TargetData& target_data = ids_iter->second;

  const auto process_direct_decorations =
      [include_linkage,
       &decorations](const std::vector<Instruction*>& direct_decorations) {
        for (Instruction* inst : direct_decorations) {
          const bool is_linkage =
              inst->opcode() == spv::Op::OpDecorate &&
              spv::Decoration(inst->GetSingleWordInOperand(1u)) ==
                  spv::Decoration::LinkageAttributes;
          if (include_linkage || !is_linkage) decorations.push_back(inst);
        }
      };

  // Process |id|'s decorations.
  process_direct_decorations(ids_iter->second.direct_decorations);

  // Process the decorations of all groups applied to |id|.
  for (const Instruction* inst : target_data.indirect_decorations) {
    const uint32_t group_id = inst->GetSingleWordInOperand(0u);
    const auto group_iter = id_to_decoration_insts_.find(group_id);
    assert(group_iter != id_to_decoration_insts_.end() && "Unknown group ID");
    process_direct_decorations(group_iter->second.direct_decorations);
  }

  return decorations;
}

bool DecorationManager::WhileEachDecoration(
    uint32_t id, uint32_t decoration,
    std::function<bool(const Instruction&)> f) const {
  for (const Instruction* inst : GetDecorationsFor(id, true)) {
    switch (inst->opcode()) {
      case spv::Op::OpMemberDecorate:
        if (inst->GetSingleWordInOperand(2) == decoration) {
          if (!f(*inst)) return false;
        }
        break;
      case spv::Op::OpDecorate:
      case spv::Op::OpDecorateId:
      case spv::Op::OpDecorateStringGOOGLE:
        if (inst->GetSingleWordInOperand(1) == decoration) {
          if (!f(*inst)) return false;
        }
        break;
      default:
        assert(false && "Unexpected decoration instruction");
    }
  }
  return true;
}

void DecorationManager::ForEachDecoration(
    uint32_t id, uint32_t decoration,
    std::function<void(const Instruction&)> f) const {
  WhileEachDecoration(id, decoration, [&f](const Instruction& inst) {
    f(inst);
    return true;
  });
}

bool DecorationManager::HasDecoration(uint32_t id,
                                      spv::Decoration decoration) const {
  return HasDecoration(id, static_cast<uint32_t>(decoration));
}

bool DecorationManager::HasDecoration(uint32_t id, uint32_t decoration) const {
  bool has_decoration = false;
  ForEachDecoration(id, decoration, [&has_decoration](const Instruction&) {
    has_decoration = true;
  });
  return has_decoration;
}

bool DecorationManager::FindDecoration(
    uint32_t id, uint32_t decoration,
    std::function<bool(const Instruction&)> f) {
  return !WhileEachDecoration(
      id, decoration, [&f](const Instruction& inst) { return !f(inst); });
}

void DecorationManager::CloneDecorations(uint32_t from, uint32_t to) {
  const auto decoration_list = id_to_decoration_insts_.find(from);
  if (decoration_list == id_to_decoration_insts_.end()) return;
  auto context = module_->context();
  for (Instruction* inst : decoration_list->second.direct_decorations) {
    // simply clone decoration and change |target-id| to |to|
    std::unique_ptr<Instruction> new_inst(inst->Clone(module_->context()));
    new_inst->SetInOperand(0, {to});
    module_->AddAnnotationInst(std::move(new_inst));
    auto decoration_iter = --module_->annotation_end();
    context->AnalyzeUses(&*decoration_iter);
  }
  // We need to copy the list of instructions as ForgetUses and AnalyzeUses are
  // going to modify it.
  std::vector<Instruction*> indirect_decorations =
      decoration_list->second.indirect_decorations;
  for (Instruction* inst : indirect_decorations) {
    switch (inst->opcode()) {
      case spv::Op::OpGroupDecorate:
        context->ForgetUses(inst);
        // add |to| to list of decorated id's
        inst->AddOperand(
            Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID, {to}));
        context->AnalyzeUses(inst);
        break;
      case spv::Op::OpGroupMemberDecorate: {
        context->ForgetUses(inst);
        // for each (id == from), add (to, literal) as operands
        const uint32_t num_operands = inst->NumOperands();
        for (uint32_t i = 1; i < num_operands; i += 2) {
          Operand op = inst->GetOperand(i);
          if (op.words[0] == from) {  // add new pair of operands: (to, literal)
            inst->AddOperand(
                Operand(spv_operand_type_t::SPV_OPERAND_TYPE_ID, {to}));
            op = inst->GetOperand(i + 1);
            inst->AddOperand(std::move(op));
          }
        }
        context->AnalyzeUses(inst);
        break;
      }
      default:
        assert(false && "Unexpected decoration instruction");
    }
  }
}

void DecorationManager::CloneDecorations(
    uint32_t from, uint32_t to,
    const std::vector<spv::Decoration>& decorations_to_copy) {
  const auto decoration_list = id_to_decoration_insts_.find(from);
  if (decoration_list == id_to_decoration_insts_.end()) return;
  auto context = module_->context();
  for (Instruction* inst : decoration_list->second.direct_decorations) {
    if (std::find(decorations_to_copy.begin(), decorations_to_copy.end(),
                  spv::Decoration(inst->GetSingleWordInOperand(1))) ==
        decorations_to_copy.end()) {
      continue;
    }

    // Clone decoration and change |target-id| to |to|.
    std::unique_ptr<Instruction> new_inst(inst->Clone(module_->context()));
    new_inst->SetInOperand(0, {to});
    module_->AddAnnotationInst(std::move(new_inst));
    auto decoration_iter = --module_->annotation_end();
    context->AnalyzeUses(&*decoration_iter);
  }

  // We need to copy the list of instructions as ForgetUses and AnalyzeUses are
  // going to modify it.
  std::vector<Instruction*> indirect_decorations =
      decoration_list->second.indirect_decorations;
  for (Instruction* inst : indirect_decorations) {
    switch (inst->opcode()) {
      case spv::Op::OpGroupDecorate:
        CloneDecorations(inst->GetSingleWordInOperand(0), to,
                         decorations_to_copy);
        break;
      case spv::Op::OpGroupMemberDecorate: {
        assert(false && "The source id is not suppose to be a type.");
        break;
      }
      default:
        assert(false && "Unexpected decoration instruction");
    }
  }
}

void DecorationManager::RemoveDecoration(Instruction* inst) {
  const auto remove_from_container = [inst](std::vector<Instruction*>& v) {
    v.erase(std::remove(v.begin(), v.end(), inst), v.end());
  };

  switch (inst->opcode()) {
    case spv::Op::OpDecorate:
    case spv::Op::OpDecorateId:
    case spv::Op::OpDecorateStringGOOGLE:
    case spv::Op::OpMemberDecorate: {
      const auto target_id = inst->GetSingleWordInOperand(0u);
      auto const iter = id_to_decoration_insts_.find(target_id);
      if (iter == id_to_decoration_insts_.end()) return;
      remove_from_container(iter->second.direct_decorations);
    } break;
    case spv::Op::OpGroupDecorate:
    case spv::Op::OpGroupMemberDecorate: {
      const uint32_t stride =
          inst->opcode() == spv::Op::OpGroupDecorate ? 1u : 2u;
      for (uint32_t i = 1u; i < inst->NumInOperands(); i += stride) {
        const auto target_id = inst->GetSingleWordInOperand(i);
        auto const iter = id_to_decoration_insts_.find(target_id);
        if (iter == id_to_decoration_insts_.end()) continue;
        remove_from_container(iter->second.indirect_decorations);
      }
      const auto group_id = inst->GetSingleWordInOperand(0u);
      auto const iter = id_to_decoration_insts_.find(group_id);
      if (iter == id_to_decoration_insts_.end()) return;
      remove_from_container(iter->second.decorate_insts);
    } break;
    default:
      break;
  }
}

bool operator==(const DecorationManager& lhs, const DecorationManager& rhs) {
  return lhs.id_to_decoration_insts_ == rhs.id_to_decoration_insts_;
}

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools
