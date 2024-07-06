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

#include "source/opt/fold_spec_constant_op_and_composite_pass.h"

#include <algorithm>
#include <tuple>

#include "source/opt/constants.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace opt {

Pass::Status FoldSpecConstantOpAndCompositePass::Process() {
  bool modified = false;
  analysis::ConstantManager* const_mgr = context()->get_constant_mgr();
  // Traverse through all the constant defining instructions. For Normal
  // Constants whose values are determined and do not depend on OpUndef
  // instructions, records their values in two internal maps: id_to_const_val_
  // and const_val_to_id_ so that we can use them to infer the value of Spec
  // Constants later.
  // For Spec Constants defined with OpSpecConstantComposite instructions, if
  // all of their components are Normal Constants, they will be turned into
  // Normal Constants too. For Spec Constants defined with OpSpecConstantOp
  // instructions, we check if they only depends on Normal Constants and fold
  // them when possible. The two maps for Normal Constants: id_to_const_val_
  // and const_val_to_id_ will be updated along the traversal so that the new
  // Normal Constants generated from folding can be used to fold following Spec
  // Constants.
  // This algorithm depends on the SSA property of SPIR-V when
  // defining constants. The dependent constants must be defined before the
  // dependee constants. So a dependent Spec Constant must be defined and
  // will be processed before its dependee Spec Constant. When we encounter
  // the dependee Spec Constants, all its dependent constants must have been
  // processed and all its dependent Spec Constants should have been folded if
  // possible.
  Module::inst_iterator next_inst = context()->types_values_begin();
  for (Module::inst_iterator inst_iter = next_inst;
       // Need to re-evaluate the end iterator since we may modify the list of
       // instructions in this section of the module as the process goes.
       inst_iter != context()->types_values_end(); inst_iter = next_inst) {
    ++next_inst;
    Instruction* inst = &*inst_iter;
    // Collect constant values of normal constants and process the
    // OpSpecConstantOp and OpSpecConstantComposite instructions if possible.
    // The constant values will be stored in analysis::Constant instances.
    // OpConstantSampler instruction is not collected here because it cannot be
    // used in OpSpecConstant{Composite|Op} instructions.
    // TODO(qining): If the constant or its type has decoration, we may need
    // to skip it.
    if (const_mgr->GetType(inst) &&
        !const_mgr->GetType(inst)->decoration_empty())
      continue;
    switch (spv::Op opcode = inst->opcode()) {
      // Records the values of Normal Constants.
      case spv::Op::OpConstantTrue:
      case spv::Op::OpConstantFalse:
      case spv::Op::OpConstant:
      case spv::Op::OpConstantNull:
      case spv::Op::OpConstantComposite:
      case spv::Op::OpSpecConstantComposite: {
        // A Constant instance will be created if the given instruction is a
        // Normal Constant whose value(s) are fixed. Note that for a composite
        // Spec Constant defined with OpSpecConstantComposite instruction, if
        // all of its components are Normal Constants already, the Spec
        // Constant will be turned in to a Normal Constant. In that case, a
        // Constant instance should also be created successfully and recorded
        // in the id_to_const_val_ and const_val_to_id_ mapps.
        if (auto const_value = const_mgr->GetConstantFromInst(inst)) {
          // Need to replace the OpSpecConstantComposite instruction with a
          // corresponding OpConstantComposite instruction.
          if (opcode == spv::Op::OpSpecConstantComposite) {
            inst->SetOpcode(spv::Op::OpConstantComposite);
            modified = true;
          }
          const_mgr->MapConstantToInst(const_value, inst);
        }
        break;
      }
      // For a Spec Constants defined with OpSpecConstantOp instruction, check
      // if it only depends on Normal Constants. If so, the Spec Constant will
      // be folded. The original Spec Constant defining instruction will be
      // replaced by Normal Constant defining instructions, and the new Normal
      // Constants will be added to id_to_const_val_ and const_val_to_id_ so
      // that we can use the new Normal Constants when folding following Spec
      // Constants.
      case spv::Op::OpSpecConstantOp:
        modified |= ProcessOpSpecConstantOp(&inst_iter);
        break;
      default:
        break;
    }
  }
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

bool FoldSpecConstantOpAndCompositePass::ProcessOpSpecConstantOp(
    Module::inst_iterator* pos) {
  Instruction* inst = &**pos;
  Instruction* folded_inst = nullptr;
  assert(inst->GetInOperand(0).type ==
             SPV_OPERAND_TYPE_SPEC_CONSTANT_OP_NUMBER &&
         "The first in-operand of OpSpecConstantOp instruction must be of "
         "SPV_OPERAND_TYPE_SPEC_CONSTANT_OP_NUMBER type");

  folded_inst = FoldWithInstructionFolder(pos);
  if (!folded_inst) {
    folded_inst = DoComponentWiseOperation(pos);
  }
  if (!folded_inst) return false;

  // Replace the original constant with the new folded constant, kill the
  // original constant.
  uint32_t new_id = folded_inst->result_id();
  uint32_t old_id = inst->result_id();
  context()->ReplaceAllUsesWith(old_id, new_id);
  context()->KillDef(old_id);
  return true;
}

Instruction* FoldSpecConstantOpAndCompositePass::FoldWithInstructionFolder(
    Module::inst_iterator* inst_iter_ptr) {
  analysis::ConstantManager* const_mgr = context()->get_constant_mgr();
  // If one of operands to the instruction is not a
  // constant, then we cannot fold this spec constant.
  for (uint32_t i = 1; i < (*inst_iter_ptr)->NumInOperands(); i++) {
    const Operand& operand = (*inst_iter_ptr)->GetInOperand(i);
    if (operand.type != SPV_OPERAND_TYPE_ID &&
        operand.type != SPV_OPERAND_TYPE_OPTIONAL_ID) {
      continue;
    }
    uint32_t id = operand.words[0];
    if (const_mgr->FindDeclaredConstant(id) == nullptr) {
      return nullptr;
    }
  }

  // All of the operands are constant.  Construct a regular version of the
  // instruction and pass it to the instruction folder.
  std::unique_ptr<Instruction> inst((*inst_iter_ptr)->Clone(context()));
  inst->SetOpcode(
      static_cast<spv::Op>((*inst_iter_ptr)->GetSingleWordInOperand(0)));
  inst->RemoveOperand(2);

  // We want the current instruction to be replaced by an |OpConstant*|
  // instruction in the same position. We need to keep track of which constants
  // the instruction folder creates, so we can move them into the correct place.
  auto last_type_value_iter = (context()->types_values_end());
  --last_type_value_iter;
  Instruction* last_type_value = &*last_type_value_iter;

  auto identity_map = [](uint32_t id) { return id; };
  Instruction* new_const_inst =
      context()->get_instruction_folder().FoldInstructionToConstant(
          inst.get(), identity_map);

  // new_const_inst == null indicates we cannot fold this spec constant
  if (!new_const_inst) return nullptr;

  // Get the instruction before |pos| to insert after.  |pos| cannot be the
  // first instruction in the list because its type has to come first.
  Instruction* insert_pos = (*inst_iter_ptr)->PreviousNode();
  assert(insert_pos != nullptr &&
         "pos is the first instruction in the types and values.");
  bool need_to_clone = true;
  for (Instruction* i = last_type_value->NextNode(); i != nullptr;
       i = last_type_value->NextNode()) {
    if (i == new_const_inst) {
      need_to_clone = false;
    }
    i->InsertAfter(insert_pos);
    insert_pos = insert_pos->NextNode();
  }

  if (need_to_clone) {
    new_const_inst = new_const_inst->Clone(context());
    new_const_inst->SetResultId(TakeNextId());
    new_const_inst->InsertAfter(insert_pos);
    get_def_use_mgr()->AnalyzeInstDefUse(new_const_inst);
  }
  const_mgr->MapInst(new_const_inst);
  return new_const_inst;
}

namespace {
// A helper function to check the type for component wise operations. Returns
// true if the type:
//  1) is bool type;
//  2) is 32-bit int type;
//  3) is vector of bool type;
//  4) is vector of 32-bit integer type.
// Otherwise returns false.
bool IsValidTypeForComponentWiseOperation(const analysis::Type* type) {
  if (type->AsBool()) {
    return true;
  } else if (auto* it = type->AsInteger()) {
    if (it->width() == 32) return true;
  } else if (auto* vt = type->AsVector()) {
    if (vt->element_type()->AsBool()) {
      return true;
    } else if (auto* vit = vt->element_type()->AsInteger()) {
      if (vit->width() == 32) return true;
    }
  }
  return false;
}

// Encodes the integer |value| of in a word vector format appropriate for
// representing this value as a operands for a constant definition. Performs
// zero-extension/sign-extension/truncation when needed, based on the signess of
// the given target type.
//
// Note: type |type| argument must be either Integer or Bool.
utils::SmallVector<uint32_t, 2> EncodeIntegerAsWords(const analysis::Type& type,
                                                     uint32_t value) {
  const uint32_t all_ones = ~0;
  uint32_t bit_width = 0;
  uint32_t pad_value = 0;
  bool result_type_signed = false;
  if (auto* int_ty = type.AsInteger()) {
    bit_width = int_ty->width();
    result_type_signed = int_ty->IsSigned();
    if (result_type_signed && static_cast<int32_t>(value) < 0) {
      pad_value = all_ones;
    }
  } else if (type.AsBool()) {
    bit_width = 1;
  } else {
    assert(false && "type must be Integer or Bool");
  }

  assert(bit_width > 0);
  uint32_t first_word = value;
  const uint32_t bits_per_word = 32;

  // Truncate first_word if the |type| has width less than uint32.
  if (bit_width < bits_per_word) {
    first_word = utils::SignExtendValue(first_word, bit_width);
  }

  utils::SmallVector<uint32_t, 2> words = {first_word};
  for (uint32_t current_bit = bits_per_word; current_bit < bit_width;
       current_bit += bits_per_word) {
    words.push_back(pad_value);
  }

  return words;
}
}  // namespace

Instruction* FoldSpecConstantOpAndCompositePass::DoComponentWiseOperation(
    Module::inst_iterator* pos) {
  const Instruction* inst = &**pos;
  analysis::ConstantManager* const_mgr = context()->get_constant_mgr();
  const analysis::Type* result_type = const_mgr->GetType(inst);
  spv::Op spec_opcode = static_cast<spv::Op>(inst->GetSingleWordInOperand(0));
  // Check and collect operands.
  std::vector<const analysis::Constant*> operands;

  if (!std::all_of(
          inst->cbegin(), inst->cend(), [&operands, this](const Operand& o) {
            // skip the operands that is not an id.
            if (o.type != spv_operand_type_t::SPV_OPERAND_TYPE_ID) return true;
            uint32_t id = o.words.front();
            if (auto c =
                    context()->get_constant_mgr()->FindDeclaredConstant(id)) {
              if (IsValidTypeForComponentWiseOperation(c->type())) {
                operands.push_back(c);
                return true;
              }
            }
            return false;
          }))
    return nullptr;

  if (result_type->AsInteger() || result_type->AsBool()) {
    // Scalar operation
    const uint32_t result_val =
        context()->get_instruction_folder().FoldScalars(spec_opcode, operands);
    auto result_const = const_mgr->GetConstant(
        result_type, EncodeIntegerAsWords(*result_type, result_val));
    return const_mgr->BuildInstructionAndAddToModule(result_const, pos);
  } else if (result_type->AsVector()) {
    // Vector operation
    const analysis::Type* element_type =
        result_type->AsVector()->element_type();
    uint32_t num_dims = result_type->AsVector()->element_count();
    std::vector<uint32_t> result_vec =
        context()->get_instruction_folder().FoldVectors(spec_opcode, num_dims,
                                                        operands);
    std::vector<const analysis::Constant*> result_vector_components;
    for (const uint32_t r : result_vec) {
      if (auto rc = const_mgr->GetConstant(
              element_type, EncodeIntegerAsWords(*element_type, r))) {
        result_vector_components.push_back(rc);
        if (!const_mgr->BuildInstructionAndAddToModule(rc, pos)) {
          assert(false &&
                 "Failed to build and insert constant declaring instruction "
                 "for the given vector component constant");
        }
      } else {
        assert(false && "Failed to create constants with 32-bit word");
      }
    }
    auto new_vec_const = MakeUnique<analysis::VectorConstant>(
        result_type->AsVector(), result_vector_components);
    auto reg_vec_const = const_mgr->RegisterConstant(std::move(new_vec_const));
    return const_mgr->BuildInstructionAndAddToModule(reg_vec_const, pos);
  } else {
    // Cannot process invalid component wise operation. The result of component
    // wise operation must be of integer or bool scalar or vector of
    // integer/bool type.
    return nullptr;
  }
}

}  // namespace opt
}  // namespace spvtools
