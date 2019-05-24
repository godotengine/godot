// Copyright (c) 2016 Google Inc.
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

#include "source/opt/def_use_manager.h"

#include <iostream>

#include "source/opt/log.h"
#include "source/opt/reflect.h"

namespace spvtools {
namespace opt {
namespace analysis {

void DefUseManager::AnalyzeInstDef(Instruction* inst) {
  const uint32_t def_id = inst->result_id();
  if (def_id != 0) {
    auto iter = id_to_def_.find(def_id);
    if (iter != id_to_def_.end()) {
      // Clear the original instruction that defining the same result id of the
      // new instruction.
      ClearInst(iter->second);
    }
    id_to_def_[def_id] = inst;
  } else {
    ClearInst(inst);
  }
}

void DefUseManager::AnalyzeInstUse(Instruction* inst) {
  // Create entry for the given instruction. Note that the instruction may
  // not have any in-operands. In such cases, we still need a entry for those
  // instructions so this manager knows it has seen the instruction later.
  auto* used_ids = &inst_to_used_ids_[inst];
  if (used_ids->size()) {
    EraseUseRecordsOfOperandIds(inst);
    used_ids = &inst_to_used_ids_[inst];
  }
  used_ids->clear();  // It might have existed before.

  for (uint32_t i = 0; i < inst->NumOperands(); ++i) {
    switch (inst->GetOperand(i).type) {
      // For any id type but result id type
      case SPV_OPERAND_TYPE_ID:
      case SPV_OPERAND_TYPE_TYPE_ID:
      case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID:
      case SPV_OPERAND_TYPE_SCOPE_ID: {
        uint32_t use_id = inst->GetSingleWordOperand(i);
        Instruction* def = GetDef(use_id);
        assert(def && "Definition is not registered.");
        id_to_users_.insert(UserEntry(def, inst));
        used_ids->push_back(use_id);
      } break;
      default:
        break;
    }
  }
}

void DefUseManager::AnalyzeInstDefUse(Instruction* inst) {
  AnalyzeInstDef(inst);
  AnalyzeInstUse(inst);
}

void DefUseManager::UpdateDefUse(Instruction* inst) {
  const uint32_t def_id = inst->result_id();
  if (def_id != 0) {
    auto iter = id_to_def_.find(def_id);
    if (iter == id_to_def_.end()) {
      AnalyzeInstDef(inst);
    }
  }
  AnalyzeInstUse(inst);
}

Instruction* DefUseManager::GetDef(uint32_t id) {
  auto iter = id_to_def_.find(id);
  if (iter == id_to_def_.end()) return nullptr;
  return iter->second;
}

const Instruction* DefUseManager::GetDef(uint32_t id) const {
  const auto iter = id_to_def_.find(id);
  if (iter == id_to_def_.end()) return nullptr;
  return iter->second;
}

DefUseManager::IdToUsersMap::const_iterator DefUseManager::UsersBegin(
    const Instruction* def) const {
  return id_to_users_.lower_bound(
      UserEntry(const_cast<Instruction*>(def), nullptr));
}

bool DefUseManager::UsersNotEnd(const IdToUsersMap::const_iterator& iter,
                                const IdToUsersMap::const_iterator& cached_end,
                                const Instruction* inst) const {
  return (iter != cached_end && iter->first == inst);
}

bool DefUseManager::UsersNotEnd(const IdToUsersMap::const_iterator& iter,
                                const Instruction* inst) const {
  return UsersNotEnd(iter, id_to_users_.end(), inst);
}

bool DefUseManager::WhileEachUser(
    const Instruction* def, const std::function<bool(Instruction*)>& f) const {
  // Ensure that |def| has been registered.
  assert(def && (!def->HasResultId() || def == GetDef(def->result_id())) &&
         "Definition is not registered.");
  if (!def->HasResultId()) return true;

  auto end = id_to_users_.end();
  for (auto iter = UsersBegin(def); UsersNotEnd(iter, end, def); ++iter) {
    if (!f(iter->second)) return false;
  }
  return true;
}

bool DefUseManager::WhileEachUser(
    uint32_t id, const std::function<bool(Instruction*)>& f) const {
  return WhileEachUser(GetDef(id), f);
}

void DefUseManager::ForEachUser(
    const Instruction* def, const std::function<void(Instruction*)>& f) const {
  WhileEachUser(def, [&f](Instruction* user) {
    f(user);
    return true;
  });
}

void DefUseManager::ForEachUser(
    uint32_t id, const std::function<void(Instruction*)>& f) const {
  ForEachUser(GetDef(id), f);
}

bool DefUseManager::WhileEachUse(
    const Instruction* def,
    const std::function<bool(Instruction*, uint32_t)>& f) const {
  // Ensure that |def| has been registered.
  assert(def && (!def->HasResultId() || def == GetDef(def->result_id())) &&
         "Definition is not registered.");
  if (!def->HasResultId()) return true;

  auto end = id_to_users_.end();
  for (auto iter = UsersBegin(def); UsersNotEnd(iter, end, def); ++iter) {
    Instruction* user = iter->second;
    for (uint32_t idx = 0; idx != user->NumOperands(); ++idx) {
      const Operand& op = user->GetOperand(idx);
      if (op.type != SPV_OPERAND_TYPE_RESULT_ID && spvIsIdType(op.type)) {
        if (def->result_id() == op.words[0]) {
          if (!f(user, idx)) return false;
        }
      }
    }
  }
  return true;
}

bool DefUseManager::WhileEachUse(
    uint32_t id, const std::function<bool(Instruction*, uint32_t)>& f) const {
  return WhileEachUse(GetDef(id), f);
}

void DefUseManager::ForEachUse(
    const Instruction* def,
    const std::function<void(Instruction*, uint32_t)>& f) const {
  WhileEachUse(def, [&f](Instruction* user, uint32_t index) {
    f(user, index);
    return true;
  });
}

void DefUseManager::ForEachUse(
    uint32_t id, const std::function<void(Instruction*, uint32_t)>& f) const {
  ForEachUse(GetDef(id), f);
}

uint32_t DefUseManager::NumUsers(const Instruction* def) const {
  uint32_t count = 0;
  ForEachUser(def, [&count](Instruction*) { ++count; });
  return count;
}

uint32_t DefUseManager::NumUsers(uint32_t id) const {
  return NumUsers(GetDef(id));
}

uint32_t DefUseManager::NumUses(const Instruction* def) const {
  uint32_t count = 0;
  ForEachUse(def, [&count](Instruction*, uint32_t) { ++count; });
  return count;
}

uint32_t DefUseManager::NumUses(uint32_t id) const {
  return NumUses(GetDef(id));
}

std::vector<Instruction*> DefUseManager::GetAnnotations(uint32_t id) const {
  std::vector<Instruction*> annos;
  const Instruction* def = GetDef(id);
  if (!def) return annos;

  ForEachUser(def, [&annos](Instruction* user) {
    if (IsAnnotationInst(user->opcode())) {
      annos.push_back(user);
    }
  });
  return annos;
}

void DefUseManager::AnalyzeDefUse(Module* module) {
  if (!module) return;
  // Analyze all the defs before any uses to catch forward references.
  module->ForEachInst(
      std::bind(&DefUseManager::AnalyzeInstDef, this, std::placeholders::_1));
  module->ForEachInst(
      std::bind(&DefUseManager::AnalyzeInstUse, this, std::placeholders::_1));
}

void DefUseManager::ClearInst(Instruction* inst) {
  auto iter = inst_to_used_ids_.find(inst);
  if (iter != inst_to_used_ids_.end()) {
    EraseUseRecordsOfOperandIds(inst);
    if (inst->result_id() != 0) {
      // Remove all uses of this inst.
      auto users_begin = UsersBegin(inst);
      auto end = id_to_users_.end();
      auto new_end = users_begin;
      for (; UsersNotEnd(new_end, end, inst); ++new_end) {
      }
      id_to_users_.erase(users_begin, new_end);
      id_to_def_.erase(inst->result_id());
    }
  }
}

void DefUseManager::EraseUseRecordsOfOperandIds(const Instruction* inst) {
  // Go through all ids used by this instruction, remove this instruction's
  // uses of them.
  auto iter = inst_to_used_ids_.find(inst);
  if (iter != inst_to_used_ids_.end()) {
    for (auto use_id : iter->second) {
      id_to_users_.erase(
          UserEntry(GetDef(use_id), const_cast<Instruction*>(inst)));
    }
    inst_to_used_ids_.erase(inst);
  }
}

bool operator==(const DefUseManager& lhs, const DefUseManager& rhs) {
  if (lhs.id_to_def_ != rhs.id_to_def_) {
    return false;
  }

  if (lhs.id_to_users_ != rhs.id_to_users_) {
    for (auto p : lhs.id_to_users_) {
      if (rhs.id_to_users_.count(p) == 0) {
        return false;
      }
    }
    for (auto p : rhs.id_to_users_) {
      if (lhs.id_to_users_.count(p) == 0) {
        return false;
      }
    }
    return false;
  }

  if (lhs.inst_to_used_ids_ != rhs.inst_to_used_ids_) {
    for (auto p : lhs.inst_to_used_ids_) {
      if (rhs.inst_to_used_ids_.count(p.first) == 0) {
        return false;
      }
    }
    for (auto p : rhs.inst_to_used_ids_) {
      if (lhs.inst_to_used_ids_.count(p.first) == 0) {
        return false;
      }
    }
    return false;
  }
  return true;
}

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools
