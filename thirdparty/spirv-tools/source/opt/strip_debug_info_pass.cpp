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

#include "source/opt/strip_debug_info_pass.h"
#include "source/opt/ir_context.h"
#include "source/util/string_utils.h"

namespace spvtools {
namespace opt {

Pass::Status StripDebugInfoPass::Process() {
  bool uses_non_semantic_info = false;
  for (auto& inst : context()->module()->extensions()) {
    const std::string ext_name = inst.GetInOperand(0).AsString();
    if (ext_name == "SPV_KHR_non_semantic_info") {
      uses_non_semantic_info = true;
    }
  }

  std::vector<Instruction*> to_kill;

  // if we use non-semantic info, it may reference OpString. Do a more
  // expensive pass checking the uses of the OpString to see if any are
  // OpExtInst on a non-semantic instruction set. If we're not using the
  // extension then we can do a simpler pass and kill all debug1 instructions
  if (uses_non_semantic_info) {
    for (auto& inst : context()->module()->debugs1()) {
      switch (inst.opcode()) {
        case spv::Op::OpString: {
          analysis::DefUseManager* def_use = context()->get_def_use_mgr();

          // see if this string is used anywhere by a non-semantic instruction
          bool no_nonsemantic_use =
              def_use->WhileEachUser(&inst, [def_use](Instruction* use) {
                if (spvIsExtendedInstruction(use->opcode())) {
                  auto ext_inst_set =
                      def_use->GetDef(use->GetSingleWordInOperand(0u));
                  const std::string extension_name =
                      ext_inst_set->GetInOperand(0).AsString();
                  if (spvtools::utils::starts_with(extension_name,
                                                   "NonSemantic.")) {
                    // found a non-semantic use, return false as we cannot
                    // remove this OpString
                    return false;
                  }
                }

                // other instructions can't be a non-semantic use
                return true;
              });

          if (no_nonsemantic_use) to_kill.push_back(&inst);

          break;
        }

        default:
          to_kill.push_back(&inst);
          break;
      }
    }
  } else {
    for (auto& dbg : context()->debugs1()) to_kill.push_back(&dbg);
  }

  for (auto& dbg : context()->debugs2()) to_kill.push_back(&dbg);
  for (auto& dbg : context()->debugs3()) to_kill.push_back(&dbg);
  for (auto& dbg : context()->ext_inst_debuginfo()) to_kill.push_back(&dbg);

  // OpName must come first, since they may refer to other debug instructions.
  // If they are after the instructions that refer to, then they will be killed
  // when that instruction is killed, which will lead to a double kill.
  std::sort(to_kill.begin(), to_kill.end(),
            [](Instruction* lhs, Instruction* rhs) -> bool {
              if (lhs->opcode() == spv::Op::OpName &&
                  rhs->opcode() != spv::Op::OpName)
                return true;
              return false;
            });

  bool modified = !to_kill.empty();

  for (auto* inst : to_kill) context()->KillInst(inst);

  // clear OpLine information
  context()->module()->ForEachInst([&modified](Instruction* inst) {
    modified |= !inst->dbg_line_insts().empty();
    inst->dbg_line_insts().clear();
  });

  if (!get_module()->trailing_dbg_line_info().empty()) {
    modified = true;
    get_module()->trailing_dbg_line_info().clear();
  }

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
