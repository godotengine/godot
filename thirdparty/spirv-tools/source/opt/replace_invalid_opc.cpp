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

  if (context()->get_feature_mgr()->HasCapability(spv::Capability::Linkage)) {
    return Status::SuccessWithoutChange;
  }

  spv::ExecutionModel execution_model = GetExecutionModel();
  if (execution_model == spv::ExecutionModel::Kernel) {
    // We do not handle kernels.
    return Status::SuccessWithoutChange;
  }
  if (execution_model == spv::ExecutionModel::Max) {
    // Mixed execution models for the entry points.  This case is not currently
    // handled.
    return Status::SuccessWithoutChange;
  }

  for (Function& func : *get_module()) {
    modified |= RewriteFunction(&func, execution_model);
  }
  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

spv::ExecutionModel ReplaceInvalidOpcodePass::GetExecutionModel() {
  spv::ExecutionModel result = spv::ExecutionModel::Max;
  bool first = true;
  for (Instruction& entry_point : get_module()->entry_points()) {
    if (first) {
      result = static_cast<spv::ExecutionModel>(
          entry_point.GetSingleWordInOperand(0));
      first = false;
    } else {
      spv::ExecutionModel current_model = static_cast<spv::ExecutionModel>(
          entry_point.GetSingleWordInOperand(0));
      if (current_model != result) {
        result = spv::ExecutionModel::Max;
        break;
      }
    }
  }
  return result;
}

bool ReplaceInvalidOpcodePass::RewriteFunction(Function* function,
                                               spv::ExecutionModel model) {
  bool modified = false;
  Instruction* last_line_dbg_inst = nullptr;
  function->ForEachInst(
      [model, &modified, &last_line_dbg_inst, this](Instruction* inst) {
        // Track the debug information so we can have a meaningful message.
        if (inst->opcode() == spv::Op::OpLabel || inst->IsNoLine()) {
          last_line_dbg_inst = nullptr;
          return;
        } else if (inst->IsLine()) {
          last_line_dbg_inst = inst;
          return;
        }

        bool replace = false;
        if (model != spv::ExecutionModel::Fragment &&
            IsFragmentShaderOnlyInstruction(inst)) {
          replace = true;
        }

        if (model != spv::ExecutionModel::TessellationControl &&
            model != spv::ExecutionModel::GLCompute) {
          if (inst->opcode() == spv::Op::OpControlBarrier) {
            assert(model != spv::ExecutionModel::Kernel &&
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
            uint32_t file_name_id = 0;
            if (last_line_dbg_inst->opcode() == spv::Op::OpLine) {
              file_name_id = last_line_dbg_inst->GetSingleWordInOperand(0);
            } else {  // Shader100::DebugLine
              uint32_t debug_source_id =
                  last_line_dbg_inst->GetSingleWordInOperand(2);
              Instruction* debug_source_inst =
                  context()->get_def_use_mgr()->GetDef(debug_source_id);
              file_name_id = debug_source_inst->GetSingleWordInOperand(2);
            }
            Instruction* file_name =
                context()->get_def_use_mgr()->GetDef(file_name_id);
            const std::string source = file_name->GetInOperand(0).AsString();

            // Get the line number and column number.
            uint32_t line_number =
                last_line_dbg_inst->GetSingleWordInOperand(1);
            uint32_t col_number = last_line_dbg_inst->GetSingleWordInOperand(2);

            // Replace the instruction.
            ReplaceInstruction(inst, source.c_str(), line_number, col_number);
          }
        }
      },
      /* run_on_debug_line_insts = */ true);
  return modified;
}

bool ReplaceInvalidOpcodePass::IsFragmentShaderOnlyInstruction(
    Instruction* inst) {
  switch (inst->opcode()) {
    case spv::Op::OpDPdx:
    case spv::Op::OpDPdy:
    case spv::Op::OpFwidth:
    case spv::Op::OpDPdxFine:
    case spv::Op::OpDPdyFine:
    case spv::Op::OpFwidthFine:
    case spv::Op::OpDPdxCoarse:
    case spv::Op::OpDPdyCoarse:
    case spv::Op::OpFwidthCoarse:
    case spv::Op::OpImageSampleImplicitLod:
    case spv::Op::OpImageSampleDrefImplicitLod:
    case spv::Op::OpImageSampleProjImplicitLod:
    case spv::Op::OpImageSampleProjDrefImplicitLod:
    case spv::Op::OpImageSparseSampleImplicitLod:
    case spv::Op::OpImageSparseSampleDrefImplicitLod:
    case spv::Op::OpImageQueryLod:
      // TODO: Teach |ReplaceInstruction| to handle block terminators.  Then
      // uncomment the OpKill case.
      // case spv::Op::OpKill:
      // case spv::Op::OpTerminateInstruction:
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
  if (type->opcode() == spv::Op::OpTypeVector) {
    uint32_t component_const =
        GetSpecialConstant(type->GetSingleWordInOperand(0));
    std::vector<uint32_t> ids;
    for (uint32_t i = 0; i < type->GetSingleWordInOperand(1); ++i) {
      ids.push_back(component_const);
    }
    special_const = const_mgr->GetConstant(type_mgr->GetType(type_id), ids);
  } else {
    assert(type->opcode() == spv::Op::OpTypeInt ||
           type->opcode() == spv::Op::OpTypeFloat);
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

std::string ReplaceInvalidOpcodePass::BuildWarningMessage(spv::Op opcode) {
  spv_opcode_desc opcode_info;
  context()->grammar().lookupOpcode(opcode, &opcode_info);
  std::string message = "Removing ";
  message += opcode_info->name;
  message += " instruction because of incompatible execution model.";
  return message;
}

}  // namespace opt
}  // namespace spvtools
