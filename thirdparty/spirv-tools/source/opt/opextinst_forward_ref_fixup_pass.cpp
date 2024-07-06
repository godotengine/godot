// Copyright (c) 2024 Google LLC
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

#include "source/opt/opextinst_forward_ref_fixup_pass.h"

#include <string>
#include <unordered_set>

#include "source/extensions.h"
#include "source/opt/ir_context.h"
#include "source/opt/module.h"
#include "type_manager.h"

namespace spvtools {
namespace opt {
namespace {

// Returns true if the instruction |inst| has a forward reference to another
// debug instruction.
// |debug_ids| contains the list of IDs belonging to debug instructions.
// |seen_ids| contains the list of IDs already seen.
bool HasForwardReference(const Instruction& inst,
                         const std::unordered_set<uint32_t>& debug_ids,
                         const std::unordered_set<uint32_t>& seen_ids) {
  const uint32_t num_in_operands = inst.NumInOperands();
  for (uint32_t i = 0; i < num_in_operands; ++i) {
    const Operand& op = inst.GetInOperand(i);
    if (!spvIsIdType(op.type)) continue;

    if (debug_ids.count(op.AsId()) == 0) continue;

    if (seen_ids.count(op.AsId()) == 0) return true;
  }

  return false;
}

// Replace |inst| opcode with OpExtInstWithForwardRefsKHR or OpExtInst
// if required to comply with forward references.
bool ReplaceOpcodeIfRequired(Instruction& inst, bool hasForwardReferences) {
  if (hasForwardReferences &&
      inst.opcode() != spv::Op::OpExtInstWithForwardRefsKHR)
    inst.SetOpcode(spv::Op::OpExtInstWithForwardRefsKHR);
  else if (!hasForwardReferences && inst.opcode() != spv::Op::OpExtInst)
    inst.SetOpcode(spv::Op::OpExtInst);
  else
    return false;
  return true;
}

// Returns all the result IDs of the instructions in |range|.
std::unordered_set<uint32_t> gatherResultIds(
    const IteratorRange<Module::inst_iterator>& range) {
  std::unordered_set<uint32_t> output;
  for (const auto& it : range) output.insert(it.result_id());
  return output;
}

}  // namespace

Pass::Status OpExtInstWithForwardReferenceFixupPass::Process() {
  std::unordered_set<uint32_t> seen_ids =
      gatherResultIds(get_module()->ext_inst_imports());
  std::unordered_set<uint32_t> debug_ids =
      gatherResultIds(get_module()->ext_inst_debuginfo());
  for (uint32_t id : seen_ids) debug_ids.insert(id);

  bool moduleChanged = false;
  bool hasAtLeastOneForwardReference = false;
  IRContext* ctx = context();
  for (Instruction& inst : get_module()->ext_inst_debuginfo()) {
    if (inst.opcode() != spv::Op::OpExtInst &&
        inst.opcode() != spv::Op::OpExtInstWithForwardRefsKHR)
      continue;

    seen_ids.insert(inst.result_id());
    bool hasForwardReferences = HasForwardReference(inst, debug_ids, seen_ids);
    hasAtLeastOneForwardReference |= hasForwardReferences;

    if (ReplaceOpcodeIfRequired(inst, hasForwardReferences)) {
      moduleChanged = true;
      ctx->AnalyzeUses(&inst);
    }
  }

  if (hasAtLeastOneForwardReference !=
      ctx->get_feature_mgr()->HasExtension(
          kSPV_KHR_relaxed_extended_instruction)) {
    if (hasAtLeastOneForwardReference)
      ctx->AddExtension("SPV_KHR_relaxed_extended_instruction");
    else
      ctx->RemoveExtension(Extension::kSPV_KHR_relaxed_extended_instruction);
    moduleChanged = true;
  }

  return moduleChanged ? Status::SuccessWithChange
                       : Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
