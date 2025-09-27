// Copyright (c) 2015-2016 The Khronos Group Inc.
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

#include "source/val/validate.h"

#include <cassert>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <stack>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "source/diagnostic.h"
#include "source/instruction.h"
#include "source/opcode.h"
#include "source/operand.h"
#include "source/spirv_validator_options.h"
#include "source/val/function.h"
#include "source/val/validation_state.h"
#include "spirv-tools/libspirv.h"

namespace spvtools {
namespace val {

spv_result_t UpdateIdUse(ValidationState_t& _, const Instruction* inst) {
  for (auto& operand : inst->operands()) {
    const spv_operand_type_t& type = operand.type;
    const uint32_t operand_id = inst->word(operand.offset);
    if (spvIsIdType(type) && type != SPV_OPERAND_TYPE_RESULT_ID) {
      if (auto def = _.FindDef(operand_id))
        def->RegisterUse(inst, operand.offset);
    }
  }

  return SPV_SUCCESS;
}

/// This function checks all ID definitions dominate their use in the CFG.
///
/// This function will iterate over all ID definitions that are defined in the
/// functions of a module and make sure that the definitions appear in a
/// block that dominates their use.
///
/// NOTE: This function does NOT check module scoped functions which are
/// checked during the initial binary parse in the IdPass below
spv_result_t CheckIdDefinitionDominateUse(ValidationState_t& _) {
  std::vector<const Instruction*> phi_instructions;
  std::unordered_set<uint32_t> phi_ids;
  for (const auto& inst : _.ordered_instructions()) {
    if (inst.id() == 0) continue;
    if (const Function* func = inst.function()) {
      if (const BasicBlock* block = inst.block()) {
        // If the Id is defined within a block then make sure all references to
        // that Id appear in a blocks that are dominated by the defining block
        for (auto& use_index_pair : inst.uses()) {
          const Instruction* use = use_index_pair.first;
          if (const BasicBlock* use_block = use->block()) {
            if (use_block->reachable() == false) continue;
            if (use->opcode() == spv::Op::OpPhi) {
              if (phi_ids.insert(use->id()).second) {
                phi_instructions.push_back(use);
              }
            } else if (!block->dominates(*use->block())) {
              return _.diag(SPV_ERROR_INVALID_ID, use_block->label())
                     << "ID " << _.getIdName(inst.id()) << " defined in block "
                     << _.getIdName(block->id())
                     << " does not dominate its use in block "
                     << _.getIdName(use_block->id());
            }
          }
        }
      } else {
        // If the Ids defined within a function but not in a block(i.e. function
        // parameters, block ids), then make sure all references to that Id
        // appear within the same function
        for (auto use : inst.uses()) {
          const Instruction* user = use.first;
          if (user->function() && user->function() != func) {
            return _.diag(SPV_ERROR_INVALID_ID, _.FindDef(func->id()))
                   << "ID " << _.getIdName(inst.id()) << " used in function "
                   << _.getIdName(user->function()->id())
                   << " is used outside of it's defining function "
                   << _.getIdName(func->id());
          }
        }
      }
    }
    // NOTE: Ids defined outside of functions must appear before they are used
    // This check is being performed in the IdPass function
  }

  // Check all OpPhi parent blocks are dominated by the variable's defining
  // blocks
  for (const Instruction* phi : phi_instructions) {
    if (phi->block()->reachable() == false) continue;
    for (size_t i = 3; i < phi->operands().size(); i += 2) {
      const Instruction* variable = _.FindDef(phi->word(i));
      const BasicBlock* parent =
          phi->function()->GetBlock(phi->word(i + 1)).first;
      if (variable->block() && parent->reachable() &&
          !variable->block()->dominates(*parent)) {
        return _.diag(SPV_ERROR_INVALID_ID, phi)
               << "In OpPhi instruction " << _.getIdName(phi->id()) << ", ID "
               << _.getIdName(variable->id())
               << " definition does not dominate its parent "
               << _.getIdName(parent->id());
      }
    }
  }

  return SPV_SUCCESS;
}

// Performs SSA validation on the IDs of an instruction. The
// can_have_forward_declared_ids  functor should return true if the
// instruction operand's ID can be forward referenced.
spv_result_t IdPass(ValidationState_t& _, Instruction* inst) {
  auto can_have_forward_declared_ids =
      inst->opcode() == spv::Op::OpExtInst &&
              spvExtInstIsDebugInfo(inst->ext_inst_type())
          ? spvDbgInfoExtOperandCanBeForwardDeclaredFunction(
                inst->ext_inst_type(), inst->word(4))
          : spvOperandCanBeForwardDeclaredFunction(inst->opcode());

  // Keep track of a result id defined by this instruction.  0 means it
  // does not define an id.
  uint32_t result_id = 0;

  for (unsigned i = 0; i < inst->operands().size(); i++) {
    const spv_parsed_operand_t& operand = inst->operand(i);
    const spv_operand_type_t& type = operand.type;
    // We only care about Id operands, which are a single word.
    const uint32_t operand_word = inst->word(operand.offset);

    auto ret = SPV_ERROR_INTERNAL;
    switch (type) {
      case SPV_OPERAND_TYPE_RESULT_ID:
        // NOTE: Multiple Id definitions are being checked by the binary parser.
        //
        // Defer undefined-forward-reference removal until after we've analyzed
        // the remaining operands to this instruction.  Deferral only matters
        // for OpPhi since it's the only case where it defines its own forward
        // reference.  Other instructions that can have forward references
        // either don't define a value or the forward reference is to a function
        // Id (and hence defined outside of a function body).
        result_id = operand_word;
        // NOTE: The result Id is added (in RegisterInstruction) *after* all of
        // the other Ids have been checked to avoid premature use in the same
        // instruction.
        ret = SPV_SUCCESS;
        break;
      case SPV_OPERAND_TYPE_ID:
      case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID:
      case SPV_OPERAND_TYPE_SCOPE_ID:
        if (const auto def = _.FindDef(operand_word)) {
          const auto opcode = inst->opcode();
          if (spvOpcodeGeneratesType(def->opcode()) &&
              !spvOpcodeGeneratesType(opcode) && !spvOpcodeIsDebug(opcode) &&
              !inst->IsDebugInfo() && !inst->IsNonSemantic() &&
              !spvOpcodeIsDecoration(opcode) && opcode != spv::Op::OpFunction &&
              opcode != spv::Op::OpCooperativeMatrixLengthNV &&
              !(opcode == spv::Op::OpSpecConstantOp &&
                spv::Op(inst->word(3)) ==
                    spv::Op::OpCooperativeMatrixLengthNV)) {
            return _.diag(SPV_ERROR_INVALID_ID, inst)
                   << "Operand " << _.getIdName(operand_word)
                   << " cannot be a type";
          } else if (def->type_id() == 0 && !spvOpcodeGeneratesType(opcode) &&
                     !spvOpcodeIsDebug(opcode) && !inst->IsDebugInfo() &&
                     !inst->IsNonSemantic() && !spvOpcodeIsDecoration(opcode) &&
                     !spvOpcodeIsBranch(opcode) && opcode != spv::Op::OpPhi &&
                     opcode != spv::Op::OpExtInst &&
                     opcode != spv::Op::OpExtInstImport &&
                     opcode != spv::Op::OpSelectionMerge &&
                     opcode != spv::Op::OpLoopMerge &&
                     opcode != spv::Op::OpFunction &&
                     opcode != spv::Op::OpCooperativeMatrixLengthNV &&
                     !(opcode == spv::Op::OpSpecConstantOp &&
                       spv::Op(inst->word(3)) ==
                           spv::Op::OpCooperativeMatrixLengthNV)) {
            return _.diag(SPV_ERROR_INVALID_ID, inst)
                   << "Operand " << _.getIdName(operand_word)
                   << " requires a type";
          } else if (def->IsNonSemantic() && !inst->IsNonSemantic()) {
            return _.diag(SPV_ERROR_INVALID_ID, inst)
                   << "Operand " << _.getIdName(operand_word)
                   << " in semantic instruction cannot be a non-semantic "
                      "instruction";
          } else {
            ret = SPV_SUCCESS;
          }
        } else if (can_have_forward_declared_ids(i)) {
          if (spvOpcodeGeneratesType(inst->opcode()) &&
              !_.IsForwardPointer(operand_word)) {
            ret = _.diag(SPV_ERROR_INVALID_ID, inst)
                  << "Operand " << _.getIdName(operand_word)
                  << " requires a previous definition";
          } else {
            ret = _.ForwardDeclareId(operand_word);
          }
        } else {
          ret = _.diag(SPV_ERROR_INVALID_ID, inst)
                << "ID " << _.getIdName(operand_word)
                << " has not been defined";
        }
        break;
      case SPV_OPERAND_TYPE_TYPE_ID:
        if (_.IsDefinedId(operand_word)) {
          auto* def = _.FindDef(operand_word);
          if (!spvOpcodeGeneratesType(def->opcode())) {
            ret = _.diag(SPV_ERROR_INVALID_ID, inst)
                  << "ID " << _.getIdName(operand_word) << " is not a type id";
          } else {
            ret = SPV_SUCCESS;
          }
        } else {
          ret = _.diag(SPV_ERROR_INVALID_ID, inst)
                << "ID " << _.getIdName(operand_word)
                << " has not been defined";
        }
        break;
      default:
        ret = SPV_SUCCESS;
        break;
    }
    if (SPV_SUCCESS != ret) return ret;
  }
  if (result_id) _.RemoveIfForwardDeclared(result_id);

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
