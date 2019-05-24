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
#include <initializer_list>
#include <tuple>

#include "source/opt/constants.h"
#include "source/opt/fold.h"
#include "source/opt/ir_context.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace opt {

Pass::Status FoldSpecConstantOpAndCompositePass::Process() {
  bool modified = false;
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
    if (context()->get_constant_mgr()->GetType(inst) &&
        !context()->get_constant_mgr()->GetType(inst)->decoration_empty())
      continue;
    switch (SpvOp opcode = inst->opcode()) {
      // Records the values of Normal Constants.
      case SpvOp::SpvOpConstantTrue:
      case SpvOp::SpvOpConstantFalse:
      case SpvOp::SpvOpConstant:
      case SpvOp::SpvOpConstantNull:
      case SpvOp::SpvOpConstantComposite:
      case SpvOp::SpvOpSpecConstantComposite: {
        // A Constant instance will be created if the given instruction is a
        // Normal Constant whose value(s) are fixed. Note that for a composite
        // Spec Constant defined with OpSpecConstantComposite instruction, if
        // all of its components are Normal Constants already, the Spec
        // Constant will be turned in to a Normal Constant. In that case, a
        // Constant instance should also be created successfully and recorded
        // in the id_to_const_val_ and const_val_to_id_ mapps.
        if (auto const_value =
                context()->get_constant_mgr()->GetConstantFromInst(inst)) {
          // Need to replace the OpSpecConstantComposite instruction with a
          // corresponding OpConstantComposite instruction.
          if (opcode == SpvOp::SpvOpSpecConstantComposite) {
            inst->SetOpcode(SpvOp::SpvOpConstantComposite);
            modified = true;
          }
          context()->get_constant_mgr()->MapConstantToInst(const_value, inst);
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
      case SpvOp::SpvOpSpecConstantOp:
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
         "The first in-operand of OpSpecContantOp instruction must be of "
         "SPV_OPERAND_TYPE_SPEC_CONSTANT_OP_NUMBER type");

  switch (static_cast<SpvOp>(inst->GetSingleWordInOperand(0))) {
    case SpvOp::SpvOpCompositeExtract:
      folded_inst = DoCompositeExtract(pos);
      break;
    case SpvOp::SpvOpVectorShuffle:
      folded_inst = DoVectorShuffle(pos);
      break;

    case SpvOp::SpvOpCompositeInsert:
      // Current Glslang does not generate code with OpSpecConstantOp
      // CompositeInsert instruction, so this is not implmented so far.
      // TODO(qining): Implement CompositeInsert case.
      return false;

    default:
      // Component-wise operations.
      folded_inst = DoComponentWiseOperation(pos);
      break;
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

uint32_t FoldSpecConstantOpAndCompositePass::GetTypeComponent(
    uint32_t typeId, uint32_t element) const {
  Instruction* type = context()->get_def_use_mgr()->GetDef(typeId);
  uint32_t subtype = type->GetTypeComponent(element);
  assert(subtype != 0);

  return subtype;
}

Instruction* FoldSpecConstantOpAndCompositePass::DoCompositeExtract(
    Module::inst_iterator* pos) {
  Instruction* inst = &**pos;
  assert(inst->NumInOperands() - 1 >= 2 &&
         "OpSpecConstantOp CompositeExtract requires at least two non-type "
         "non-opcode operands.");
  assert(inst->GetInOperand(1).type == SPV_OPERAND_TYPE_ID &&
         "The composite operand must have a SPV_OPERAND_TYPE_ID type");
  assert(
      inst->GetInOperand(2).type == SPV_OPERAND_TYPE_LITERAL_INTEGER &&
      "The literal operand must have a SPV_OPERAND_TYPE_LITERAL_INTEGER type");

  // Note that for OpSpecConstantOp, the second in-operand is the first id
  // operand. The first in-operand is the spec opcode.
  uint32_t source = inst->GetSingleWordInOperand(1);
  uint32_t type = context()->get_def_use_mgr()->GetDef(source)->type_id();
  const analysis::Constant* first_operand_const =
      context()->get_constant_mgr()->FindDeclaredConstant(source);
  if (!first_operand_const) return nullptr;

  const analysis::Constant* current_const = first_operand_const;
  for (uint32_t i = 2; i < inst->NumInOperands(); i++) {
    uint32_t literal = inst->GetSingleWordInOperand(i);
    type = GetTypeComponent(type, literal);
  }
  for (uint32_t i = 2; i < inst->NumInOperands(); i++) {
    uint32_t literal = inst->GetSingleWordInOperand(i);
    if (const analysis::CompositeConstant* composite_const =
            current_const->AsCompositeConstant()) {
      // Case 1: current constant is a non-null composite type constant.
      assert(literal < composite_const->GetComponents().size() &&
             "Literal index out of bound of the composite constant");
      current_const = composite_const->GetComponents().at(literal);
    } else if (current_const->AsNullConstant()) {
      // Case 2: current constant is a constant created with OpConstantNull.
      // Because components of a NullConstant are always NullConstants, we can
      // return early with a NullConstant in the result type.
      return context()->get_constant_mgr()->BuildInstructionAndAddToModule(
          context()->get_constant_mgr()->GetConstant(
              context()->get_constant_mgr()->GetType(inst), {}),
          pos, type);
    } else {
      // Dereferencing a non-composite constant. Invalid case.
      return nullptr;
    }
  }
  return context()->get_constant_mgr()->BuildInstructionAndAddToModule(
      current_const, pos);
}

Instruction* FoldSpecConstantOpAndCompositePass::DoVectorShuffle(
    Module::inst_iterator* pos) {
  Instruction* inst = &**pos;
  analysis::Vector* result_vec_type =
      context()->get_constant_mgr()->GetType(inst)->AsVector();
  assert(inst->NumInOperands() - 1 > 2 &&
         "OpSpecConstantOp DoVectorShuffle instruction requires more than 2 "
         "operands (2 vector ids and at least one literal operand");
  assert(result_vec_type &&
         "The result of VectorShuffle must be of type vector");

  // A temporary null constants that can be used as the components of the result
  // vector. This is needed when any one of the vector operands are null
  // constant.
  const analysis::Constant* null_component_constants = nullptr;

  // Get a concatenated vector of scalar constants. The vector should be built
  // with the components from the first and the second operand of VectorShuffle.
  std::vector<const analysis::Constant*> concatenated_components;
  // Note that for OpSpecConstantOp, the second in-operand is the first id
  // operand. The first in-operand is the spec opcode.
  for (uint32_t i : {1, 2}) {
    assert(inst->GetInOperand(i).type == SPV_OPERAND_TYPE_ID &&
           "The vector operand must have a SPV_OPERAND_TYPE_ID type");
    uint32_t operand_id = inst->GetSingleWordInOperand(i);
    auto operand_const =
        context()->get_constant_mgr()->FindDeclaredConstant(operand_id);
    if (!operand_const) return nullptr;
    const analysis::Type* operand_type = operand_const->type();
    assert(operand_type->AsVector() &&
           "The first two operand of VectorShuffle must be of vector type");
    if (auto vec_const = operand_const->AsVectorConstant()) {
      // case 1: current operand is a non-null vector constant.
      concatenated_components.insert(concatenated_components.end(),
                                     vec_const->GetComponents().begin(),
                                     vec_const->GetComponents().end());
    } else if (operand_const->AsNullConstant()) {
      // case 2: current operand is a null vector constant. Create a temporary
      // null scalar constant as the component.
      if (!null_component_constants) {
        const analysis::Type* component_type =
            operand_type->AsVector()->element_type();
        null_component_constants =
            context()->get_constant_mgr()->GetConstant(component_type, {});
      }
      // Append the null scalar consts to the concatenated components
      // vector.
      concatenated_components.insert(concatenated_components.end(),
                                     operand_type->AsVector()->element_count(),
                                     null_component_constants);
    } else {
      // no other valid cases
      return nullptr;
    }
  }
  // Create null component constants if there are any. The component constants
  // must be added to the module before the dependee composite constants to
  // satisfy SSA def-use dominance.
  if (null_component_constants) {
    context()->get_constant_mgr()->BuildInstructionAndAddToModule(
        null_component_constants, pos);
  }
  // Create the new vector constant with the selected components.
  std::vector<const analysis::Constant*> selected_components;
  for (uint32_t i = 3; i < inst->NumInOperands(); i++) {
    assert(inst->GetInOperand(i).type == SPV_OPERAND_TYPE_LITERAL_INTEGER &&
           "The literal operand must of type SPV_OPERAND_TYPE_LITERAL_INTEGER");
    uint32_t literal = inst->GetSingleWordInOperand(i);
    assert(literal < concatenated_components.size() &&
           "Literal index out of bound of the concatenated vector");
    selected_components.push_back(concatenated_components[literal]);
  }
  auto new_vec_const = MakeUnique<analysis::VectorConstant>(
      result_vec_type, selected_components);
  auto reg_vec_const =
      context()->get_constant_mgr()->RegisterConstant(std::move(new_vec_const));
  return context()->get_constant_mgr()->BuildInstructionAndAddToModule(
      reg_vec_const, pos);
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
}  // namespace

Instruction* FoldSpecConstantOpAndCompositePass::DoComponentWiseOperation(
    Module::inst_iterator* pos) {
  const Instruction* inst = &**pos;
  const analysis::Type* result_type =
      context()->get_constant_mgr()->GetType(inst);
  SpvOp spec_opcode = static_cast<SpvOp>(inst->GetSingleWordInOperand(0));
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
    uint32_t result_val =
        context()->get_instruction_folder().FoldScalars(spec_opcode, operands);
    auto result_const =
        context()->get_constant_mgr()->GetConstant(result_type, {result_val});
    return context()->get_constant_mgr()->BuildInstructionAndAddToModule(
        result_const, pos);
  } else if (result_type->AsVector()) {
    // Vector operation
    const analysis::Type* element_type =
        result_type->AsVector()->element_type();
    uint32_t num_dims = result_type->AsVector()->element_count();
    std::vector<uint32_t> result_vec =
        context()->get_instruction_folder().FoldVectors(spec_opcode, num_dims,
                                                        operands);
    std::vector<const analysis::Constant*> result_vector_components;
    for (uint32_t r : result_vec) {
      if (auto rc =
              context()->get_constant_mgr()->GetConstant(element_type, {r})) {
        result_vector_components.push_back(rc);
        if (!context()->get_constant_mgr()->BuildInstructionAndAddToModule(
                rc, pos)) {
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
    auto reg_vec_const = context()->get_constant_mgr()->RegisterConstant(
        std::move(new_vec_const));
    return context()->get_constant_mgr()->BuildInstructionAndAddToModule(
        reg_vec_const, pos);
  } else {
    // Cannot process invalid component wise operation. The result of component
    // wise operation must be of integer or bool scalar or vector of
    // integer/bool type.
    return nullptr;
  }
}

}  // namespace opt
}  // namespace spvtools
