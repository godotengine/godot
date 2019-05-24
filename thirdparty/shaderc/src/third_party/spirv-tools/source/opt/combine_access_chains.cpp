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

#include "source/opt/combine_access_chains.h"

#include <utility>

#include "source/opt/constants.h"
#include "source/opt/ir_builder.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {

Pass::Status CombineAccessChains::Process() {
  bool modified = false;

  for (auto& function : *get_module()) {
    modified |= ProcessFunction(function);
  }

  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

bool CombineAccessChains::ProcessFunction(Function& function) {
  bool modified = false;

  cfg()->ForEachBlockInReversePostOrder(
      function.entry().get(), [&modified, this](BasicBlock* block) {
        block->ForEachInst([&modified, this](Instruction* inst) {
          switch (inst->opcode()) {
            case SpvOpAccessChain:
            case SpvOpInBoundsAccessChain:
            case SpvOpPtrAccessChain:
            case SpvOpInBoundsPtrAccessChain:
              modified |= CombineAccessChain(inst);
              break;
            default:
              break;
          }
        });
      });

  return modified;
}

uint32_t CombineAccessChains::GetConstantValue(
    const analysis::Constant* constant_inst) {
  if (constant_inst->type()->AsInteger()->width() <= 32) {
    if (constant_inst->type()->AsInteger()->IsSigned()) {
      return static_cast<uint32_t>(constant_inst->GetS32());
    } else {
      return constant_inst->GetU32();
    }
  } else {
    assert(false);
    return 0u;
  }
}

uint32_t CombineAccessChains::GetArrayStride(const Instruction* inst) {
  uint32_t array_stride = 0;
  context()->get_decoration_mgr()->WhileEachDecoration(
      inst->type_id(), SpvDecorationArrayStride,
      [&array_stride](const Instruction& decoration) {
        assert(decoration.opcode() != SpvOpDecorateId);
        if (decoration.opcode() == SpvOpDecorate) {
          array_stride = decoration.GetSingleWordInOperand(1);
        } else {
          array_stride = decoration.GetSingleWordInOperand(2);
        }
        return false;
      });
  return array_stride;
}

const analysis::Type* CombineAccessChains::GetIndexedType(Instruction* inst) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  analysis::TypeManager* type_mgr = context()->get_type_mgr();

  Instruction* base_ptr = def_use_mgr->GetDef(inst->GetSingleWordInOperand(0));
  const analysis::Type* type = type_mgr->GetType(base_ptr->type_id());
  assert(type->AsPointer());
  type = type->AsPointer()->pointee_type();
  std::vector<uint32_t> element_indices;
  uint32_t starting_index = 1;
  if (IsPtrAccessChain(inst->opcode())) {
    // Skip the first index of OpPtrAccessChain as it does not affect type
    // resolution.
    starting_index = 2;
  }
  for (uint32_t i = starting_index; i < inst->NumInOperands(); ++i) {
    Instruction* index_inst =
        def_use_mgr->GetDef(inst->GetSingleWordInOperand(i));
    const analysis::Constant* index_constant =
        context()->get_constant_mgr()->GetConstantFromInst(index_inst);
    if (index_constant) {
      uint32_t index_value = GetConstantValue(index_constant);
      element_indices.push_back(index_value);
    } else {
      // This index must not matter to resolve the type in valid SPIR-V.
      element_indices.push_back(0);
    }
  }
  type = type_mgr->GetMemberType(type, element_indices);
  return type;
}

bool CombineAccessChains::CombineIndices(Instruction* ptr_input,
                                         Instruction* inst,
                                         std::vector<Operand>* new_operands) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  analysis::ConstantManager* constant_mgr = context()->get_constant_mgr();

  Instruction* last_index_inst = def_use_mgr->GetDef(
      ptr_input->GetSingleWordInOperand(ptr_input->NumInOperands() - 1));
  const analysis::Constant* last_index_constant =
      constant_mgr->GetConstantFromInst(last_index_inst);

  Instruction* element_inst =
      def_use_mgr->GetDef(inst->GetSingleWordInOperand(1));
  const analysis::Constant* element_constant =
      constant_mgr->GetConstantFromInst(element_inst);

  // Combine the last index of the AccessChain (|ptr_inst|) with the element
  // operand of the PtrAccessChain (|inst|).
  const bool combining_element_operands =
      IsPtrAccessChain(inst->opcode()) &&
      IsPtrAccessChain(ptr_input->opcode()) && ptr_input->NumInOperands() == 2;
  uint32_t new_value_id = 0;
  const analysis::Type* type = GetIndexedType(ptr_input);
  if (last_index_constant && element_constant) {
    // Combine the constants.
    uint32_t new_value = GetConstantValue(last_index_constant) +
                         GetConstantValue(element_constant);
    const analysis::Constant* new_value_constant =
        constant_mgr->GetConstant(last_index_constant->type(), {new_value});
    Instruction* new_value_inst =
        constant_mgr->GetDefiningInstruction(new_value_constant);
    new_value_id = new_value_inst->result_id();
  } else if (!type->AsStruct() || combining_element_operands) {
    // Generate an addition of the two indices.
    InstructionBuilder builder(
        context(), inst,
        IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
    Instruction* addition = builder.AddIAdd(last_index_inst->type_id(),
                                            last_index_inst->result_id(),
                                            element_inst->result_id());
    new_value_id = addition->result_id();
  } else {
    // Indexing into structs must be constant, so bail out here.
    return false;
  }
  new_operands->push_back({SPV_OPERAND_TYPE_ID, {new_value_id}});
  return true;
}

bool CombineAccessChains::CreateNewInputOperands(
    Instruction* ptr_input, Instruction* inst,
    std::vector<Operand>* new_operands) {
  // Start by copying all the input operands of the feeder access chain.
  for (uint32_t i = 0; i != ptr_input->NumInOperands() - 1; ++i) {
    new_operands->push_back(ptr_input->GetInOperand(i));
  }

  // Deal with the last index of the feeder access chain.
  if (IsPtrAccessChain(inst->opcode())) {
    // The last index of the feeder should be combined with the element operand
    // of |inst|.
    if (!CombineIndices(ptr_input, inst, new_operands)) return false;
  } else {
    // The indices aren't being combined so now add the last index operand of
    // |ptr_input|.
    new_operands->push_back(
        ptr_input->GetInOperand(ptr_input->NumInOperands() - 1));
  }

  // Copy the remaining index operands.
  uint32_t starting_index = IsPtrAccessChain(inst->opcode()) ? 2 : 1;
  for (uint32_t i = starting_index; i < inst->NumInOperands(); ++i) {
    new_operands->push_back(inst->GetInOperand(i));
  }

  return true;
}

bool CombineAccessChains::CombineAccessChain(Instruction* inst) {
  assert((inst->opcode() == SpvOpPtrAccessChain ||
          inst->opcode() == SpvOpAccessChain ||
          inst->opcode() == SpvOpInBoundsAccessChain ||
          inst->opcode() == SpvOpInBoundsPtrAccessChain) &&
         "Wrong opcode. Expected an access chain.");

  Instruction* ptr_input =
      context()->get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(0));
  if (ptr_input->opcode() != SpvOpAccessChain &&
      ptr_input->opcode() != SpvOpInBoundsAccessChain &&
      ptr_input->opcode() != SpvOpPtrAccessChain &&
      ptr_input->opcode() != SpvOpInBoundsPtrAccessChain) {
    return false;
  }

  if (Has64BitIndices(inst) || Has64BitIndices(ptr_input)) return false;

  // Handles the following cases:
  // 1. |ptr_input| is an index-less access chain. Replace the pointer
  //    in |inst| with |ptr_input|'s pointer.
  // 2. |inst| is a index-less access chain. Change |inst| to an
  //    OpCopyObject.
  // 3. |inst| is not a pointer access chain.
  //    |inst|'s indices are appended to |ptr_input|'s indices.
  // 4. |ptr_input| is not pointer access chain.
  //    |inst| is a pointer access chain.
  //    |inst|'s element operand is combined with the last index in
  //    |ptr_input| to form a new operand.
  // 5. |ptr_input| is a pointer access chain.
  //    Like the above scenario, |inst|'s element operand is combined
  //    with |ptr_input|'s last index. This results is either a
  //    combined element operand or combined regular index.

  // TODO(alan-baker): Support this properly. Requires analyzing the
  // size/alignment of the type and converting the stride into an element
  // index.
  uint32_t array_stride = GetArrayStride(ptr_input);
  if (array_stride != 0) return false;

  if (ptr_input->NumInOperands() == 1) {
    // The input is effectively a no-op.
    inst->SetInOperand(0, {ptr_input->GetSingleWordInOperand(0)});
    context()->AnalyzeUses(inst);
  } else if (inst->NumInOperands() == 1) {
    // |inst| is a no-op, change it to a copy. Instruction simplification will
    // clean it up.
    inst->SetOpcode(SpvOpCopyObject);
  } else {
    std::vector<Operand> new_operands;
    if (!CreateNewInputOperands(ptr_input, inst, &new_operands)) return false;

    // Update the instruction.
    inst->SetOpcode(UpdateOpcode(inst->opcode(), ptr_input->opcode()));
    inst->SetInOperands(std::move(new_operands));
    context()->AnalyzeUses(inst);
  }
  return true;
}

SpvOp CombineAccessChains::UpdateOpcode(SpvOp base_opcode, SpvOp input_opcode) {
  auto IsInBounds = [](SpvOp opcode) {
    return opcode == SpvOpInBoundsPtrAccessChain ||
           opcode == SpvOpInBoundsAccessChain;
  };

  if (input_opcode == SpvOpInBoundsPtrAccessChain) {
    if (!IsInBounds(base_opcode)) return SpvOpPtrAccessChain;
  } else if (input_opcode == SpvOpInBoundsAccessChain) {
    if (!IsInBounds(base_opcode)) return SpvOpAccessChain;
  }

  return input_opcode;
}

bool CombineAccessChains::IsPtrAccessChain(SpvOp opcode) {
  return opcode == SpvOpPtrAccessChain || opcode == SpvOpInBoundsPtrAccessChain;
}

bool CombineAccessChains::Has64BitIndices(Instruction* inst) {
  for (uint32_t i = 1; i < inst->NumInOperands(); ++i) {
    Instruction* index_inst =
        context()->get_def_use_mgr()->GetDef(inst->GetSingleWordInOperand(i));
    const analysis::Type* index_type =
        context()->get_type_mgr()->GetType(index_inst->type_id());
    if (!index_type->AsInteger() || index_type->AsInteger()->width() != 32)
      return true;
  }
  return false;
}

}  // namespace opt
}  // namespace spvtools
