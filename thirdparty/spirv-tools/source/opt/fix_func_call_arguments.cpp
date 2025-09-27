// Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#include "fix_func_call_arguments.h"

#include "ir_builder.h"

using namespace spvtools;
using namespace opt;

bool FixFuncCallArgumentsPass::ModuleHasASingleFunction() {
  auto funcsNum = get_module()->end() - get_module()->begin();
  return funcsNum == 1;
}

Pass::Status FixFuncCallArgumentsPass::Process() {
  bool modified = false;
  if (ModuleHasASingleFunction()) return Status::SuccessWithoutChange;
  for (auto& func : *get_module()) {
    func.ForEachInst([this, &modified](Instruction* inst) {
      if (inst->opcode() == spv::Op::OpFunctionCall) {
        modified |= FixFuncCallArguments(inst);
      }
    });
  }
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

bool FixFuncCallArgumentsPass::FixFuncCallArguments(
    Instruction* func_call_inst) {
  bool modified = false;
  for (uint32_t i = 0; i < func_call_inst->NumInOperands(); ++i) {
    Operand& op = func_call_inst->GetInOperand(i);
    if (op.type != SPV_OPERAND_TYPE_ID) continue;
    Instruction* operand_inst = get_def_use_mgr()->GetDef(op.AsId());
    if (operand_inst->opcode() == spv::Op::OpAccessChain) {
      uint32_t var_id =
          ReplaceAccessChainFuncCallArguments(func_call_inst, operand_inst);
      func_call_inst->SetInOperand(i, {var_id});
      modified = true;
    }
  }
  if (modified) {
    context()->UpdateDefUse(func_call_inst);
  }
  return modified;
}

uint32_t FixFuncCallArgumentsPass::ReplaceAccessChainFuncCallArguments(
    Instruction* func_call_inst, Instruction* operand_inst) {
  InstructionBuilder builder(
      context(), func_call_inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);

  Instruction* next_insert_point = func_call_inst->NextNode();
  // Get Variable insertion point
  Function* func = context()->get_instr_block(func_call_inst)->GetParent();
  Instruction* variable_insertion_point = &*(func->begin()->begin());
  Instruction* op_ptr_type = get_def_use_mgr()->GetDef(operand_inst->type_id());
  Instruction* op_type =
      get_def_use_mgr()->GetDef(op_ptr_type->GetSingleWordInOperand(1));
  uint32_t varType = context()->get_type_mgr()->FindPointerToType(
      op_type->result_id(), spv::StorageClass::Function);
  // Create new variable
  builder.SetInsertPoint(variable_insertion_point);
  Instruction* var =
      builder.AddVariable(varType, uint32_t(spv::StorageClass::Function));
  // Load access chain to the new variable before function call
  builder.SetInsertPoint(func_call_inst);

  uint32_t operand_id = operand_inst->result_id();
  Instruction* load = builder.AddLoad(op_type->result_id(), operand_id);
  builder.AddStore(var->result_id(), load->result_id());
  // Load return value to the acesschain after function call
  builder.SetInsertPoint(next_insert_point);
  load = builder.AddLoad(op_type->result_id(), var->result_id());
  builder.AddStore(operand_id, load->result_id());

  return var->result_id();
}
