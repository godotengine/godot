// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
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

#include "source/opt/inline_opaque_pass.h"

#include <utility>

namespace spvtools {
namespace opt {
namespace {

const uint32_t kTypePointerTypeIdInIdx = 1;

}  // anonymous namespace

bool InlineOpaquePass::IsOpaqueType(uint32_t typeId) {
  const Instruction* typeInst = get_def_use_mgr()->GetDef(typeId);
  switch (typeInst->opcode()) {
    case SpvOpTypeSampler:
    case SpvOpTypeImage:
    case SpvOpTypeSampledImage:
      return true;
    case SpvOpTypePointer:
      return IsOpaqueType(
          typeInst->GetSingleWordInOperand(kTypePointerTypeIdInIdx));
    default:
      break;
  }
  // TODO(greg-lunarg): Handle arrays containing opaque type
  if (typeInst->opcode() != SpvOpTypeStruct) return false;
  // Return true if any member is opaque
  return !typeInst->WhileEachInId([this](const uint32_t* tid) {
    if (IsOpaqueType(*tid)) return false;
    return true;
  });
}

bool InlineOpaquePass::HasOpaqueArgsOrReturn(const Instruction* callInst) {
  // Check return type
  if (IsOpaqueType(callInst->type_id())) return true;
  // Check args
  int icnt = 0;
  return !callInst->WhileEachInId([&icnt, this](const uint32_t* iid) {
    if (icnt > 0) {
      const Instruction* argInst = get_def_use_mgr()->GetDef(*iid);
      if (IsOpaqueType(argInst->type_id())) return false;
    }
    ++icnt;
    return true;
  });
}

Pass::Status InlineOpaquePass::InlineOpaque(Function* func) {
  bool modified = false;
  // Using block iterators here because of block erasures and insertions.
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end();) {
      if (IsInlinableFunctionCall(&*ii) && HasOpaqueArgsOrReturn(&*ii)) {
        // Inline call.
        std::vector<std::unique_ptr<BasicBlock>> newBlocks;
        std::vector<std::unique_ptr<Instruction>> newVars;
        if (!GenInlineCode(&newBlocks, &newVars, ii, bi)) {
          return Status::Failure;
        }

        // If call block is replaced with more than one block, point
        // succeeding phis at new last block.
        if (newBlocks.size() > 1) UpdateSucceedingPhis(newBlocks);
        // Replace old calling block with new block(s).
        bi = bi.Erase();
        bi = bi.InsertBefore(&newBlocks);
        // Insert new function variables.
        if (newVars.size() > 0)
          func->begin()->begin().InsertBefore(std::move(newVars));
        // Restart inlining at beginning of calling block.
        ii = bi->begin();
        modified = true;
      } else {
        ++ii;
      }
    }
  }
  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

void InlineOpaquePass::Initialize() { InitializeInline(); }

Pass::Status InlineOpaquePass::ProcessImpl() {
  Status status = Status::SuccessWithoutChange;
  // Do opaque inlining on each function in entry point call tree
  ProcessFunction pfn = [&status, this](Function* fp) {
    status = CombineStatus(status, InlineOpaque(fp));
    return false;
  };
  context()->ProcessEntryPointCallTree(pfn);
  return status;
}

InlineOpaquePass::InlineOpaquePass() = default;

Pass::Status InlineOpaquePass::Process() {
  Initialize();
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools
