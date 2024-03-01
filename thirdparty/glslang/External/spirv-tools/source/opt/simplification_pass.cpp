// Copyright (c) 2018 Google LLC
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

#include "source/opt/simplification_pass.h"

#include <unordered_set>
#include <vector>

#include "source/opt/fold.h"

namespace spvtools {
namespace opt {

Pass::Status SimplificationPass::Process() {
  bool modified = false;

  for (Function& function : *get_module()) {
    modified |= SimplifyFunction(&function);
  }
  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

void SimplificationPass::AddNewOperands(
    Instruction* folded_inst, std::unordered_set<Instruction*>* inst_seen,
    std::vector<Instruction*>* work_list) {
  analysis::DefUseManager* def_use_mgr = get_def_use_mgr();
  folded_inst->ForEachInId(
      [&inst_seen, &def_use_mgr, &work_list](uint32_t* iid) {
        Instruction* iid_inst = def_use_mgr->GetDef(*iid);
        if (!inst_seen->insert(iid_inst).second) return;
        work_list->push_back(iid_inst);
      });
}

bool SimplificationPass::SimplifyFunction(Function* function) {
  if (function->IsDeclaration()) {
    return false;
  }

  bool modified = false;
  // Phase 1: Traverse all instructions in dominance order.
  // The second phase will only be on the instructions whose inputs have changed
  // after being processed during phase 1.  Since OpPhi instructions are the
  // only instructions whose inputs do not necessarily dominate the use, we keep
  // track of the OpPhi instructions already seen, and add them to the work list
  // for phase 2 when needed.
  std::vector<Instruction*> work_list;
  std::unordered_set<Instruction*> process_phis;
  std::unordered_set<Instruction*> inst_to_kill;
  std::unordered_set<Instruction*> in_work_list;
  std::unordered_set<Instruction*> inst_seen;
  const InstructionFolder& folder = context()->get_instruction_folder();

  cfg()->ForEachBlockInReversePostOrder(
      function->entry().get(),
      [&modified, &process_phis, &work_list, &in_work_list, &inst_to_kill,
       &folder, &inst_seen, this](BasicBlock* bb) {
        for (Instruction* inst = &*bb->begin(); inst; inst = inst->NextNode()) {
          inst_seen.insert(inst);
          if (inst->opcode() == spv::Op::OpPhi) {
            process_phis.insert(inst);
          }

          bool is_foldable_copy =
              inst->opcode() == spv::Op::OpCopyObject &&
              context()->get_decoration_mgr()->HaveSubsetOfDecorations(
                  inst->result_id(), inst->GetSingleWordInOperand(0));

          if (is_foldable_copy || folder.FoldInstruction(inst)) {
            modified = true;
            context()->AnalyzeUses(inst);
            get_def_use_mgr()->ForEachUser(inst, [&work_list, &process_phis,
                                                  &in_work_list](
                                                     Instruction* use) {
              if (process_phis.count(use) && in_work_list.insert(use).second) {
                work_list.push_back(use);
              }
            });

            AddNewOperands(inst, &inst_seen, &work_list);

            if (inst->opcode() == spv::Op::OpCopyObject) {
              context()->ReplaceAllUsesWithPredicate(
                  inst->result_id(), inst->GetSingleWordInOperand(0),
                  [](Instruction* user) {
                    const auto opcode = user->opcode();
                    if (!spvOpcodeIsDebug(opcode) &&
                        !spvOpcodeIsDecoration(opcode)) {
                      return true;
                    }
                    return false;
                  });
              inst_to_kill.insert(inst);
              in_work_list.insert(inst);
            } else if (inst->opcode() == spv::Op::OpNop) {
              inst_to_kill.insert(inst);
              in_work_list.insert(inst);
            }
          }
        }
      });

  // Phase 2: process the instructions in the work list until all of the work is
  //          done.  This time we add all users to the work list because phase 1
  //          has already finished.
  for (size_t i = 0; i < work_list.size(); ++i) {
    Instruction* inst = work_list[i];
    in_work_list.erase(inst);
    inst_seen.insert(inst);

    bool is_foldable_copy =
        inst->opcode() == spv::Op::OpCopyObject &&
        context()->get_decoration_mgr()->HaveSubsetOfDecorations(
            inst->result_id(), inst->GetSingleWordInOperand(0));

    if (is_foldable_copy || folder.FoldInstruction(inst)) {
      modified = true;
      context()->AnalyzeUses(inst);
      get_def_use_mgr()->ForEachUser(
          inst, [&work_list, &in_work_list](Instruction* use) {
            if (!use->IsDecoration() && use->opcode() != spv::Op::OpName &&
                in_work_list.insert(use).second) {
              work_list.push_back(use);
            }
          });

      AddNewOperands(inst, &inst_seen, &work_list);

      if (inst->opcode() == spv::Op::OpCopyObject) {
        context()->ReplaceAllUsesWithPredicate(
            inst->result_id(), inst->GetSingleWordInOperand(0),
            [](Instruction* user) {
              const auto opcode = user->opcode();
              if (!spvOpcodeIsDebug(opcode) && !spvOpcodeIsDecoration(opcode)) {
                return true;
              }
              return false;
            });
        inst_to_kill.insert(inst);
        in_work_list.insert(inst);
      } else if (inst->opcode() == spv::Op::OpNop) {
        inst_to_kill.insert(inst);
        in_work_list.insert(inst);
      }
    }
  }

  // Phase 3: Kill instructions we know are no longer needed.
  for (Instruction* inst : inst_to_kill) {
    context()->KillInst(inst);
  }

  return modified;
}

}  // namespace opt
}  // namespace spvtools
