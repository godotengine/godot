// Copyright (c) 2019 Google LLC
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

#include "source/opt/wrap_opkill.h"

#include "ir_builder.h"

namespace spvtools {
namespace opt {

Pass::Status WrapOpKill::Process() {
  bool modified = false;

  auto func_to_process =
      context()->GetStructuredCFGAnalysis()->FindFuncsCalledFromContinue();
  for (uint32_t func_id : func_to_process) {
    Function* func = context()->GetFunction(func_id);
    bool successful = func->WhileEachInst([this, &modified](Instruction* inst) {
      const auto opcode = inst->opcode();
      if ((opcode == spv::Op::OpKill) ||
          (opcode == spv::Op::OpTerminateInvocation)) {
        modified = true;
        if (!ReplaceWithFunctionCall(inst)) {
          return false;
        }
      }
      return true;
    });

    if (!successful) {
      return Status::Failure;
    }
  }

  if (opkill_function_ != nullptr) {
    assert(modified &&
           "The function should only be generated if something was modified.");
    context()->AddFunction(std::move(opkill_function_));
  }
  if (opterminateinvocation_function_ != nullptr) {
    assert(modified &&
           "The function should only be generated if something was modified.");
    context()->AddFunction(std::move(opterminateinvocation_function_));
  }
  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

bool WrapOpKill::ReplaceWithFunctionCall(Instruction* inst) {
  assert((inst->opcode() == spv::Op::OpKill ||
          inst->opcode() == spv::Op::OpTerminateInvocation) &&
         "|inst| must be an OpKill or OpTerminateInvocation instruction.");
  InstructionBuilder ir_builder(
      context(), inst,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  uint32_t func_id = GetKillingFuncId(inst->opcode());
  if (func_id == 0) {
    return false;
  }
  Instruction* call_inst =
      ir_builder.AddFunctionCall(GetVoidTypeId(), func_id, {});
  if (call_inst == nullptr) {
    return false;
  }
  call_inst->UpdateDebugInfoFrom(inst);

  Instruction* return_inst = nullptr;
  uint32_t return_type_id = GetOwningFunctionsReturnType(inst);
  if (return_type_id != GetVoidTypeId()) {
    Instruction* undef =
        ir_builder.AddNullaryOp(return_type_id, spv::Op::OpUndef);
    if (undef == nullptr) {
      return false;
    }
    return_inst =
        ir_builder.AddUnaryOp(0, spv::Op::OpReturnValue, undef->result_id());
  } else {
    return_inst = ir_builder.AddNullaryOp(0, spv::Op::OpReturn);
  }

  if (return_inst == nullptr) {
    return false;
  }

  context()->KillInst(inst);
  return true;
}

uint32_t WrapOpKill::GetVoidTypeId() {
  if (void_type_id_ != 0) {
    return void_type_id_;
  }

  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::Void void_type;
  void_type_id_ = type_mgr->GetTypeInstruction(&void_type);
  return void_type_id_;
}

uint32_t WrapOpKill::GetVoidFunctionTypeId() {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::Void void_type;
  const analysis::Type* registered_void_type =
      type_mgr->GetRegisteredType(&void_type);

  analysis::Function func_type(registered_void_type, {});
  return type_mgr->GetTypeInstruction(&func_type);
}

uint32_t WrapOpKill::GetKillingFuncId(spv::Op opcode) {
  //  Parameterize by opcode
  assert(opcode == spv::Op::OpKill || opcode == spv::Op::OpTerminateInvocation);

  std::unique_ptr<Function>* const killing_func =
      (opcode == spv::Op::OpKill) ? &opkill_function_
                                  : &opterminateinvocation_function_;

  if (*killing_func != nullptr) {
    return (*killing_func)->result_id();
  }

  uint32_t killing_func_id = TakeNextId();
  if (killing_func_id == 0) {
    return 0;
  }

  uint32_t void_type_id = GetVoidTypeId();
  if (void_type_id == 0) {
    return 0;
  }

  // Generate the function start instruction
  std::unique_ptr<Instruction> func_start(new Instruction(
      context(), spv::Op::OpFunction, void_type_id, killing_func_id, {}));
  func_start->AddOperand({SPV_OPERAND_TYPE_FUNCTION_CONTROL, {0}});
  func_start->AddOperand({SPV_OPERAND_TYPE_ID, {GetVoidFunctionTypeId()}});
  (*killing_func).reset(new Function(std::move(func_start)));

  // Generate the function end instruction
  std::unique_ptr<Instruction> func_end(
      new Instruction(context(), spv::Op::OpFunctionEnd, 0, 0, {}));
  (*killing_func)->SetFunctionEnd(std::move(func_end));

  // Create the one basic block for the function.
  uint32_t lab_id = TakeNextId();
  if (lab_id == 0) {
    return 0;
  }
  std::unique_ptr<Instruction> label_inst(
      new Instruction(context(), spv::Op::OpLabel, 0, lab_id, {}));
  std::unique_ptr<BasicBlock> bb(new BasicBlock(std::move(label_inst)));

  // Add the OpKill to the basic block
  std::unique_ptr<Instruction> kill_inst(
      new Instruction(context(), opcode, 0, 0, {}));
  bb->AddInstruction(std::move(kill_inst));

  // Add the bb to the function
  (*killing_func)->AddBasicBlock(std::move(bb));

  // Add the function to the module.
  if (context()->AreAnalysesValid(IRContext::kAnalysisDefUse)) {
    (*killing_func)->ForEachInst([this](Instruction* inst) {
      context()->AnalyzeDefUse(inst);
    });
  }

  if (context()->AreAnalysesValid(IRContext::kAnalysisInstrToBlockMapping)) {
    for (BasicBlock& basic_block : *(*killing_func)) {
      context()->set_instr_block(basic_block.GetLabelInst(), &basic_block);
      for (Instruction& inst : basic_block) {
        context()->set_instr_block(&inst, &basic_block);
      }
    }
  }

  return (*killing_func)->result_id();
}

uint32_t WrapOpKill::GetOwningFunctionsReturnType(Instruction* inst) {
  BasicBlock* bb = context()->get_instr_block(inst);
  if (bb == nullptr) {
    return 0;
  }

  Function* func = bb->GetParent();
  return func->type_id();
}

}  // namespace opt
}  // namespace spvtools
