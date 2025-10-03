// Copyright (c) 2018 The Khronos Group Inc.
// Copyright (c) 2018 Valve Corporation
// Copyright (c) 2018 LunarG Inc.
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

#include "source/opt/dead_insert_elim_pass.h"

#include "source/opt/composite.h"
#include "source/opt/ir_context.h"
#include "source/opt/iterator.h"
#include "spirv/1.2/GLSL.std.450.h"

namespace spvtools {
namespace opt {
namespace {
constexpr uint32_t kTypeVectorCountInIdx = 1;
constexpr uint32_t kTypeMatrixCountInIdx = 1;
constexpr uint32_t kTypeArrayLengthIdInIdx = 1;
constexpr uint32_t kTypeIntWidthInIdx = 0;
constexpr uint32_t kConstantValueInIdx = 0;
constexpr uint32_t kInsertObjectIdInIdx = 0;
constexpr uint32_t kInsertCompositeIdInIdx = 1;
}  // namespace

uint32_t DeadInsertElimPass::NumComponents(Instruction* typeInst) {
  switch (typeInst->opcode()) {
    case spv::Op::OpTypeVector: {
      return typeInst->GetSingleWordInOperand(kTypeVectorCountInIdx);
    } break;
    case spv::Op::OpTypeMatrix: {
      return typeInst->GetSingleWordInOperand(kTypeMatrixCountInIdx);
    } break;
    case spv::Op::OpTypeArray: {
      uint32_t lenId =
          typeInst->GetSingleWordInOperand(kTypeArrayLengthIdInIdx);
      Instruction* lenInst = get_def_use_mgr()->GetDef(lenId);
      if (lenInst->opcode() != spv::Op::OpConstant) return 0;
      uint32_t lenTypeId = lenInst->type_id();
      Instruction* lenTypeInst = get_def_use_mgr()->GetDef(lenTypeId);
      // TODO(greg-lunarg): Support non-32-bit array length
      if (lenTypeInst->GetSingleWordInOperand(kTypeIntWidthInIdx) != 32)
        return 0;
      return lenInst->GetSingleWordInOperand(kConstantValueInIdx);
    } break;
    case spv::Op::OpTypeStruct: {
      return typeInst->NumInOperands();
    } break;
    default: { return 0; } break;
  }
}

void DeadInsertElimPass::MarkInsertChain(
    Instruction* insertChain, std::vector<uint32_t>* pExtIndices,
    uint32_t extOffset, std::unordered_set<uint32_t>* visited_phis) {
  // Not currently optimizing array inserts.
  Instruction* typeInst = get_def_use_mgr()->GetDef(insertChain->type_id());
  if (typeInst->opcode() == spv::Op::OpTypeArray) return;
  // Insert chains are only composed of inserts and phis
  if (insertChain->opcode() != spv::Op::OpCompositeInsert &&
      insertChain->opcode() != spv::Op::OpPhi)
    return;
  // If extract indices are empty, mark all subcomponents if type
  // is constant length.
  if (pExtIndices == nullptr) {
    uint32_t cnum = NumComponents(typeInst);
    if (cnum > 0) {
      std::vector<uint32_t> extIndices;
      for (uint32_t i = 0; i < cnum; i++) {
        extIndices.clear();
        extIndices.push_back(i);
        std::unordered_set<uint32_t> sub_visited_phis;
        MarkInsertChain(insertChain, &extIndices, 0, &sub_visited_phis);
      }
      return;
    }
  }
  Instruction* insInst = insertChain;
  while (insInst->opcode() == spv::Op::OpCompositeInsert) {
    // If no extract indices, mark insert and inserted object (which might
    // also be an insert chain) and continue up the chain though the input
    // composite.
    //
    // Note: We mark inserted objects in this function (rather than in
    // EliminateDeadInsertsOnePass) because in some cases, we can do it
    // more accurately here.
    if (pExtIndices == nullptr) {
      liveInserts_.insert(insInst->result_id());
      uint32_t objId = insInst->GetSingleWordInOperand(kInsertObjectIdInIdx);
      std::unordered_set<uint32_t> obj_visited_phis;
      MarkInsertChain(get_def_use_mgr()->GetDef(objId), nullptr, 0,
                      &obj_visited_phis);
    // If extract indices match insert, we are done. Mark insert and
    // inserted object.
    } else if (ExtInsMatch(*pExtIndices, insInst, extOffset)) {
      liveInserts_.insert(insInst->result_id());
      uint32_t objId = insInst->GetSingleWordInOperand(kInsertObjectIdInIdx);
      std::unordered_set<uint32_t> obj_visited_phis;
      MarkInsertChain(get_def_use_mgr()->GetDef(objId), nullptr, 0,
                      &obj_visited_phis);
      break;
    // If non-matching intersection, mark insert
    } else if (ExtInsConflict(*pExtIndices, insInst, extOffset)) {
      liveInserts_.insert(insInst->result_id());
      // If more extract indices than insert, we are done. Use remaining
      // extract indices to mark inserted object.
      uint32_t numInsertIndices = insInst->NumInOperands() - 2;
      if (pExtIndices->size() - extOffset > numInsertIndices) {
        uint32_t objId = insInst->GetSingleWordInOperand(kInsertObjectIdInIdx);
        std::unordered_set<uint32_t> obj_visited_phis;
        MarkInsertChain(get_def_use_mgr()->GetDef(objId), pExtIndices,
                        extOffset + numInsertIndices, &obj_visited_phis);
        break;
      // If fewer extract indices than insert, also mark inserted object and
      // continue up chain.
      } else {
        uint32_t objId = insInst->GetSingleWordInOperand(kInsertObjectIdInIdx);
        std::unordered_set<uint32_t> obj_visited_phis;
        MarkInsertChain(get_def_use_mgr()->GetDef(objId), nullptr, 0,
                        &obj_visited_phis);
      }
    }
    // Get next insert in chain
    const uint32_t compId =
        insInst->GetSingleWordInOperand(kInsertCompositeIdInIdx);
    insInst = get_def_use_mgr()->GetDef(compId);
  }
  // If insert chain ended with phi, do recursive call on each operand
  if (insInst->opcode() != spv::Op::OpPhi) return;
  // Mark phi visited to prevent potential infinite loop. If phi is already
  // visited, return to avoid infinite loop.
  if (visited_phis->count(insInst->result_id()) != 0) return;
  visited_phis->insert(insInst->result_id());

  // Phis may have duplicate inputs values for different edges, prune incoming
  // ids lists before recursing.
  std::vector<uint32_t> ids;
  for (uint32_t i = 0; i < insInst->NumInOperands(); i += 2) {
    ids.push_back(insInst->GetSingleWordInOperand(i));
  }
  std::sort(ids.begin(), ids.end());
  auto new_end = std::unique(ids.begin(), ids.end());
  for (auto id_iter = ids.begin(); id_iter != new_end; ++id_iter) {
    Instruction* pi = get_def_use_mgr()->GetDef(*id_iter);
    MarkInsertChain(pi, pExtIndices, extOffset, visited_phis);
  }
}

bool DeadInsertElimPass::EliminateDeadInserts(Function* func) {
  bool modified = false;
  bool lastmodified = true;
  // Each pass can delete dead instructions, thus potentially revealing
  // new dead insertions ie insertions with no uses.
  while (lastmodified) {
    lastmodified = EliminateDeadInsertsOnePass(func);
    modified |= lastmodified;
  }
  return modified;
}

bool DeadInsertElimPass::EliminateDeadInsertsOnePass(Function* func) {
  bool modified = false;
  liveInserts_.clear();
  visitedPhis_.clear();
  // Mark all live inserts
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      // Only process Inserts and composite Phis
      spv::Op op = ii->opcode();
      Instruction* typeInst = get_def_use_mgr()->GetDef(ii->type_id());
      if (op != spv::Op::OpCompositeInsert &&
          (op != spv::Op::OpPhi || !spvOpcodeIsComposite(typeInst->opcode())))
        continue;
      // The marking algorithm can be expensive for large arrays and the
      // efficacy of eliminating dead inserts into arrays is questionable.
      // Skip optimizing array inserts for now. Just mark them live.
      // TODO(greg-lunarg): Eliminate dead array inserts
      if (op == spv::Op::OpCompositeInsert) {
        if (typeInst->opcode() == spv::Op::OpTypeArray) {
          liveInserts_.insert(ii->result_id());
          continue;
        }
      }
      const uint32_t id = ii->result_id();
      get_def_use_mgr()->ForEachUser(id, [&ii, this](Instruction* user) {
        if (user->IsCommonDebugInstr()) return;
        switch (user->opcode()) {
          case spv::Op::OpCompositeInsert:
          case spv::Op::OpPhi:
            // Use by insert or phi does not initiate marking
            break;
          case spv::Op::OpCompositeExtract: {
            // Capture extract indices
            std::vector<uint32_t> extIndices;
            uint32_t icnt = 0;
            user->ForEachInOperand([&icnt, &extIndices](const uint32_t* idp) {
              if (icnt > 0) extIndices.push_back(*idp);
              ++icnt;
            });
            // Mark all inserts in chain that intersect with extract
            std::unordered_set<uint32_t> visited_phis;
            MarkInsertChain(&*ii, &extIndices, 0, &visited_phis);
          } break;
          default: {
            // Mark inserts in chain for all components
            MarkInsertChain(&*ii, nullptr, 0, nullptr);
          } break;
        }
      });
    }
  }
  // Find and disconnect dead inserts
  std::vector<Instruction*> dead_instructions;
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end(); ++ii) {
      if (ii->opcode() != spv::Op::OpCompositeInsert) continue;
      const uint32_t id = ii->result_id();
      if (liveInserts_.find(id) != liveInserts_.end()) continue;
      const uint32_t replId =
          ii->GetSingleWordInOperand(kInsertCompositeIdInIdx);
      (void)context()->ReplaceAllUsesWith(id, replId);
      dead_instructions.push_back(&*ii);
      modified = true;
    }
  }
  // DCE dead inserts
  while (!dead_instructions.empty()) {
    Instruction* inst = dead_instructions.back();
    dead_instructions.pop_back();
    DCEInst(inst, [&dead_instructions](Instruction* other_inst) {
      auto i = std::find(dead_instructions.begin(), dead_instructions.end(),
                         other_inst);
      if (i != dead_instructions.end()) {
        dead_instructions.erase(i);
      }
    });
  }
  return modified;
}

Pass::Status DeadInsertElimPass::Process() {
  // Process all entry point functions.
  ProcessFunction pfn = [this](Function* fp) {
    return EliminateDeadInserts(fp);
  };
  bool modified = context()->ProcessReachableCallTree(pfn);
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
