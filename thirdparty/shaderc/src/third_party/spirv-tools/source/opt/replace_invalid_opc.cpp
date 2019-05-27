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

#include "source/opt/replace_invalid_opc.h"

#include <bitset>
#include <vector>

namespace spvtools {
namespace opt {

Pass::Status ReplaceInvalidOpcodePass::Process() {
  bool modified = false;

  if (context()->get_feature_mgr()->HasCapability(SpvCapabilityLinkage)) {
    return Status::SuccessWithoutChange;
  }

  SpvExecutionModel execution_model = GetExecutionModel();
  if (execution_model == SpvExecutionModelKernel) {
    // We do not handle kernels.
    return Status::SuccessWithoutChange;
  }
  if (execution_model == SpvExecutionModelMax) {
    // Mixed execution models for the entry points.  This case is not currently
    // handled.
    return Status::SuccessWithoutChange;
  }

  for (Function& func : *get_module()) {
    modified |= RewriteFunction(&func, execution_model);
  }
  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

SpvExecutionModel ReplaceInvalidOpcodePass::GetExecutionModel() {
  SpvExecutionModel result = SpvExecutionModelMax;
  bool first = true;
  for (Instruction& entry_point : get_module()->entry_points()) {
    if (first) {
      result =
          static_cast<SpvExecutionModel>(entry_point.GetSingleWordInOperand(0));
      first = false;
    } else {
      SpvExecutionModel current_model =
          static_cast<SpvExecutionModel>(entry_point.GetSingleWordInOperand(0));
      if (current_model != result) {
        result = SpvExecutionModelMax;
        break;
      }
    }
  }
  return result;
}

bool ReplaceInvalidOpcodePass::RewriteFunction(Function* function,
                                               SpvExecutionModel model) {
  bool modified = false;
  Instruction* last_line_dbg_inst = nullptr;
  function->ForEachInst(
      [model, &modified, &last_line_dbg_inst, this](Instruction* inst) {
        // Track the debug information so we can have a meaningful message.
        if (inst->opcode() == SpvOpLabel || inst->opcode() == SpvOpNoLine) {
          last_line_dbg_inst = nullptr;
          return;
        } else if (inst->opcode() == SpvOpLine) {
          last_line_dbg_inst = inst;
          return;
        }

        bool replace = false;
        if (model != SpvExecutionModelFragment &&
            IsFragmentShaderOnlyInstruction(inst)) {
          replace = true;
        }

        if (model != SpvExecutionModelTessellationControl &&
            model != SpvExecutionModelGLCompute) {
          if (inst->opcode() == SpvOpControlBarrier) {
            assert(model != SpvExecutionModelKernel &&
                   "Expecting to be working on a shader module.");
            replace = true;
          }
        }

        if (replace) {
          modified = true;
          if (last_line_dbg_inst == nullptr) {
            ReplaceInstruction(inst, nullptr, 0, 0);
          } else {
            // Get the name of the source file.
            Instruction* file_name = context()->get_def_use_mgr()->GetDef(
                last_line_dbg_inst->GetSingleWordInOperand(0));
            const char* source = reinterpret_cast<const char*>(
                &file_name->GetInOperand(0).words[0]);

            // Get the line number and column number.
            uint32_t line_number =
                last_line_dbg_inst->GetSingleWordInOperand(1);
            uint32_t col_number = last_line_dbg_inst->GetSingleWordInOperand(2);

            // Replace the instruction.
            ReplaceInstruction(inst, source, line_number, col_number);
          }
        }
      },
      /* run_on_debug_line_insts = */ true);
  return modified;
}

bool ReplaceInvalidOpcodePass::IsFragmentShaderOnlyInstruction(
    Instruction* inst) {
  switch (inst->opcode()) {
    case SpvOpDPdx:
    case SpvOpDPdy:
    case SpvOpFwidth:
    case SpvOpDPdxFine:
    case SpvOpDPdyFine:
    case SpvOpFwidthFine:
    case SpvOpDPdxCoarse:
    case SpvOpDPdyCoarse:
    case SpvOpFwidthCoarse:
    case SpvOpImageSampleImplicitLod:
    case SpvOpImageSampleDrefImplicitLod:
    case SpvOpImageSampleProjImplicitLod:
    case SpvOpImageSampleProjDrefImplicitLod:
    case SpvOpImageSparseSampleImplicitLod:
    case SpvOpImageSparseSampleDrefImplicitLod:
    case SpvOpImageQueryLod:
      // TODO: Teach |ReplaceInstruction| to handle block terminators.  Then
      // uncomment the OpKill case.
      // case SpvOpKill:
      return true;
    default:
      return false;
  }
}

void ReplaceInvalidOpcodePass::ReplaceInstruction(Instruction* inst,
                                                  const char* source,
                                                  uint32_t line_number,
                                                  uint32_t column_number) {
  if (inst->result_id() != 0) {
    uint32_t const_id = GetSpecialConstant(inst->type_id());
    context()->KillNamesAndDecorates(inst);
    context()->ReplaceAllUsesWith(inst->result_id(), const_id);
  }
  assert(!inst->IsBlockTerminator() &&
         "We cannot simply delete a block terminator.  It must be replaced "
         "with something.");
  if (consumer()) {
    std::string message = BuildWarningMessage(inst->opcode());
    consumer()(SPV_MSG_WARNING, source, {line_number, column_number, 0},
               message.c_str());
  }
  context()->KillInst(inst);
}

uint32_t ReplaceInvalidOpcodePass::GetSpecialConstant(uint32_t type_id) {
  const analysis::Constant* special_const = nullptr;
  analysis::ConstantManager* const_mgr = context()->get_constant_mgr();
  analysis::TypeManager* type_mgr = context()->get_type_mgr();

  Instruction* type = context()->get_def_use_mgr()->GetDef(type_id);
  if (type->opcode() == SpvOpTypeVector) {
    uint32_t component_const =
        GetSpecialConstant(type->GetSingleWordInOperand(0));
    std::vector<uint32_t> ids;
    for (uint32_t i = 0; i < type->GetSingleWordInOperand(1); ++i) {
      ids.push_back(component_const);
    }
    special_const = const_mgr->GetConstant(type_mgr->GetType(type_id), ids);
  } else {
    assert(type->opcode() == SpvOpTypeInt || type->opcode() == SpvOpTypeFloat);
    std::vector<uint32_t> literal_words;
    for (uint32_t i = 0; i < type->GetSingleWordInOperand(0); i += 32) {
      literal_words.push_back(0xDEADBEEF);
    }
    special_const =
        const_mgr->GetConstant(type_mgr->GetType(type_id), literal_words);
  }
  assert(special_const != nullptr);
  return const_mgr->GetDefiningInstruction(special_const)->result_id();
}

std::string ReplaceInvalidOpcodePass::BuildWarningMessage(SpvOp opcode) {
  spv_opcode_desc opcode_info;
  context()->grammar().lookupOpcode(opcode, &opcode_info);
  std::string message = "Removing ";
  message += opcode_info->name;
  message += " instruction because of incompatible execution model.";
  return message;
}

}  // namespace opt
}  // namespace spvtools
