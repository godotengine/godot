// Copyright (c) 2017 Google Inc.
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

// This file implements conditional constant propagation as described in
//
//      Constant propagation with conditional branches,
//      Wegman and Zadeck, ACM TOPLAS 13(2):181-210.

#include "source/opt/ccp_pass.h"

#include <algorithm>
#include <limits>

#include "source/opt/fold.h"
#include "source/opt/function.h"
#include "source/opt/propagator.h"

namespace spvtools {
namespace opt {
namespace {
// This SSA id is never defined nor referenced in the IR.  It is a special ID
// which represents varying values.  When an ID is found to have a varying
// value, its entry in the |values_| table maps to kVaryingSSAId.
constexpr uint32_t kVaryingSSAId = std::numeric_limits<uint32_t>::max();
}  // namespace

bool CCPPass::IsVaryingValue(uint32_t id) const { return id == kVaryingSSAId; }

SSAPropagator::PropStatus CCPPass::MarkInstructionVarying(Instruction* instr) {
  assert(instr->result_id() != 0 &&
         "Instructions with no result cannot be marked varying.");
  values_[instr->result_id()] = kVaryingSSAId;
  return SSAPropagator::kVarying;
}

SSAPropagator::PropStatus CCPPass::VisitPhi(Instruction* phi) {
  uint32_t meet_val_id = 0;

  // Implement the lattice meet operation. The result of this Phi instruction is
  // interesting only if the meet operation over arguments coming through
  // executable edges yields the same constant value.
  for (uint32_t i = 2; i < phi->NumOperands(); i += 2) {
    if (!propagator_->IsPhiArgExecutable(phi, i)) {
      // Ignore arguments coming through non-executable edges.
      continue;
    }
    uint32_t phi_arg_id = phi->GetSingleWordOperand(i);
    auto it = values_.find(phi_arg_id);
    if (it != values_.end()) {
      // We found an argument with a constant value.  Apply the meet operation
      // with the previous arguments.
      if (it->second == kVaryingSSAId) {
        // The "constant" value is actually a placeholder for varying. Return
        // varying for this phi.
        return MarkInstructionVarying(phi);
      } else if (meet_val_id == 0) {
        // This is the first argument we find.  Initialize the result to its
        // constant value id.
        meet_val_id = it->second;
      } else if (it->second == meet_val_id) {
        // The argument is the same constant value already computed. Continue
        // looking.
        continue;
      } else {
        // We either found a varying value, or another constant value different
        // from the previous computed meet value.  This Phi will never be
        // constant.
        return MarkInstructionVarying(phi);
      }
    } else {
      // The incoming value has no recorded value and is therefore not
      // interesting. A not interesting value joined with any other value is the
      // other value.
      continue;
    }
  }

  // If there are no incoming executable edges, the meet ID will still be 0. In
  // that case, return not interesting to evaluate the Phi node again.
  if (meet_val_id == 0) {
    return SSAPropagator::kNotInteresting;
  }

  // All the operands have the same constant value represented by |meet_val_id|.
  // Set the Phi's result to that value and declare it interesting.
  values_[phi->result_id()] = meet_val_id;
  return SSAPropagator::kInteresting;
}

uint32_t CCPPass::ComputeLatticeMeet(Instruction* instr, uint32_t val2) {
  // Given two values val1 and val2, the meet operation in the constant
  // lattice uses the following rules:
  //
  // meet(val1, UNDEFINED) = val1
  // meet(val1, VARYING)   = VARYING
  // meet(val1, val2)      = val1     if val1 == val2
  // meet(val1, val2)      = VARYING  if val1 != val2
  //
  // When two different values meet, the result is always varying because CCP
  // does not allow lateral transitions in the lattice.  This prevents
  // infinite cycles during propagation.
  auto val1_it = values_.find(instr->result_id());
  if (val1_it == values_.end()) {
    return val2;
  }

  uint32_t val1 = val1_it->second;
  if (IsVaryingValue(val1)) {
    return val1;
  } else if (IsVaryingValue(val2)) {
    return val2;
  } else if (val1 != val2) {
    return kVaryingSSAId;
  }
  return val2;
}

SSAPropagator::PropStatus CCPPass::VisitAssignment(Instruction* instr) {
  assert(instr->result_id() != 0 &&
         "Expecting an instruction that produces a result");

  // If this is a copy operation, and the RHS is a known constant, assign its
  // value to the LHS.
  if (instr->opcode() == spv::Op::OpCopyObject) {
    uint32_t rhs_id = instr->GetSingleWordInOperand(0);
    auto it = values_.find(rhs_id);
    if (it != values_.end()) {
      if (IsVaryingValue(it->second)) {
        return MarkInstructionVarying(instr);
      } else {
        uint32_t new_val = ComputeLatticeMeet(instr, it->second);
        values_[instr->result_id()] = new_val;
        return IsVaryingValue(new_val) ? SSAPropagator::kVarying
                                       : SSAPropagator::kInteresting;
      }
    }
    return SSAPropagator::kNotInteresting;
  }

  // Instructions with a RHS that cannot produce a constant are always varying.
  if (!instr->IsFoldable()) {
    return MarkInstructionVarying(instr);
  }

  // See if the RHS of the assignment folds into a constant value.
  auto map_func = [this](uint32_t id) {
    auto it = values_.find(id);
    if (it == values_.end() || IsVaryingValue(it->second)) {
      return id;
    }
    return it->second;
  };
  Instruction* folded_inst =
      context()->get_instruction_folder().FoldInstructionToConstant(instr,
                                                                    map_func);

  if (folded_inst != nullptr) {
    // We do not want to change the body of the function by adding new
    // instructions.  When folding we can only generate new constants.
    assert((folded_inst->IsConstant() ||
            IsSpecConstantInst(folded_inst->opcode())) &&
           "CCP is only interested in constant values.");
    uint32_t new_val = ComputeLatticeMeet(instr, folded_inst->result_id());
    values_[instr->result_id()] = new_val;
    return IsVaryingValue(new_val) ? SSAPropagator::kVarying
                                   : SSAPropagator::kInteresting;
  }

  // Conservatively mark this instruction as varying if any input id is varying.
  if (!instr->WhileEachInId([this](uint32_t* op_id) {
        auto iter = values_.find(*op_id);
        if (iter != values_.end() && IsVaryingValue(iter->second)) return false;
        return true;
      })) {
    return MarkInstructionVarying(instr);
  }

  // If not, see if there is a least one unknown operand to the instruction.  If
  // so, we might be able to fold it later.
  if (!instr->WhileEachInId([this](uint32_t* op_id) {
        auto it = values_.find(*op_id);
        if (it == values_.end()) return false;
        return true;
      })) {
    return SSAPropagator::kNotInteresting;
  }

  // Otherwise, we will never be able to fold this instruction, so mark it
  // varying.
  return MarkInstructionVarying(instr);
}

SSAPropagator::PropStatus CCPPass::VisitBranch(Instruction* instr,
                                               BasicBlock** dest_bb) const {
  assert(instr->IsBranch() && "Expected a branch instruction.");

  *dest_bb = nullptr;
  uint32_t dest_label = 0;
  if (instr->opcode() == spv::Op::OpBranch) {
    // An unconditional jump always goes to its unique destination.
    dest_label = instr->GetSingleWordInOperand(0);
  } else if (instr->opcode() == spv::Op::OpBranchConditional) {
    // For a conditional branch, determine whether the predicate selector has a
    // known value in |values_|.  If it does, set the destination block
    // according to the selector's boolean value.
    uint32_t pred_id = instr->GetSingleWordOperand(0);
    auto it = values_.find(pred_id);
    if (it == values_.end() || IsVaryingValue(it->second)) {
      // The predicate has an unknown value, either branch could be taken.
      return SSAPropagator::kVarying;
    }

    // Get the constant value for the predicate selector from the value table.
    // Use it to decide which branch will be taken.
    uint32_t pred_val_id = it->second;
    const analysis::Constant* c = const_mgr_->FindDeclaredConstant(pred_val_id);
    assert(c && "Expected to find a constant declaration for a known value.");
    // Undef values should have returned as varying above.
    assert(c->AsBoolConstant() || c->AsNullConstant());
    if (c->AsNullConstant()) {
      dest_label = instr->GetSingleWordOperand(2u);
    } else {
      const analysis::BoolConstant* val = c->AsBoolConstant();
      dest_label = val->value() ? instr->GetSingleWordOperand(1)
                                : instr->GetSingleWordOperand(2);
    }
  } else {
    // For an OpSwitch, extract the value taken by the switch selector and check
    // which of the target literals it matches.  The branch associated with that
    // literal is the taken branch.
    assert(instr->opcode() == spv::Op::OpSwitch);
    if (instr->GetOperand(0).words.size() != 1) {
      // If the selector is wider than 32-bits, return varying. TODO(dnovillo):
      // Add support for wider constants.
      return SSAPropagator::kVarying;
    }
    uint32_t select_id = instr->GetSingleWordOperand(0);
    auto it = values_.find(select_id);
    if (it == values_.end() || IsVaryingValue(it->second)) {
      // The selector has an unknown value, any of the branches could be taken.
      return SSAPropagator::kVarying;
    }

    // Get the constant value for the selector from the value table. Use it to
    // decide which branch will be taken.
    uint32_t select_val_id = it->second;
    const analysis::Constant* c =
        const_mgr_->FindDeclaredConstant(select_val_id);
    assert(c && "Expected to find a constant declaration for a known value.");
    // TODO: support 64-bit integer switches.
    uint32_t constant_cond = 0;
    if (const analysis::IntConstant* val = c->AsIntConstant()) {
      constant_cond = val->words()[0];
    } else {
      // Undef values should have returned varying above.
      assert(c->AsNullConstant());
      constant_cond = 0;
    }

    // Start assuming that the selector will take the default value;
    dest_label = instr->GetSingleWordOperand(1);
    for (uint32_t i = 2; i < instr->NumOperands(); i += 2) {
      if (constant_cond == instr->GetSingleWordOperand(i)) {
        dest_label = instr->GetSingleWordOperand(i + 1);
        break;
      }
    }
  }

  assert(dest_label && "Destination label should be set at this point.");
  *dest_bb = context()->cfg()->block(dest_label);
  return SSAPropagator::kInteresting;
}

SSAPropagator::PropStatus CCPPass::VisitInstruction(Instruction* instr,
                                                    BasicBlock** dest_bb) {
  *dest_bb = nullptr;
  if (instr->opcode() == spv::Op::OpPhi) {
    return VisitPhi(instr);
  } else if (instr->IsBranch()) {
    return VisitBranch(instr, dest_bb);
  } else if (instr->result_id()) {
    return VisitAssignment(instr);
  }
  return SSAPropagator::kVarying;
}

bool CCPPass::ReplaceValues() {
  // Even if we make no changes to the function's IR, propagation may have
  // created new constants.  Even if those constants cannot be replaced in
  // the IR, the constant definition itself is a change.  To reflect this,
  // we check whether the next ID to be given by the module is different than
  // the original bound ID. If that happens, new instructions were added to the
  // module during propagation.
  //
  // See https://github.com/KhronosGroup/SPIRV-Tools/issues/3636 and
  // https://github.com/KhronosGroup/SPIRV-Tools/issues/3991 for details.
  bool changed_ir = (context()->module()->IdBound() > original_id_bound_);

  for (const auto& it : values_) {
    uint32_t id = it.first;
    uint32_t cst_id = it.second;
    if (!IsVaryingValue(cst_id) && id != cst_id) {
      context()->KillNamesAndDecorates(id);
      changed_ir |= context()->ReplaceAllUsesWith(id, cst_id);
    }
  }

  return changed_ir;
}

bool CCPPass::PropagateConstants(Function* fp) {
  if (fp->IsDeclaration()) {
    return false;
  }

  // Mark function parameters as varying.
  fp->ForEachParam([this](const Instruction* inst) {
    values_[inst->result_id()] = kVaryingSSAId;
  });

  const auto visit_fn = [this](Instruction* instr, BasicBlock** dest_bb) {
    return VisitInstruction(instr, dest_bb);
  };

  propagator_ =
      std::unique_ptr<SSAPropagator>(new SSAPropagator(context(), visit_fn));

  if (propagator_->Run(fp)) {
    return ReplaceValues();
  }

  return false;
}

void CCPPass::Initialize() {
  const_mgr_ = context()->get_constant_mgr();

  // Populate the constant table with values from constant declarations in the
  // module.  The values of each OpConstant declaration is the identity
  // assignment (i.e., each constant is its own value).
  for (const auto& inst : get_module()->types_values()) {
    // Record compile time constant ids. Treat all other global values as
    // varying.
    if (inst.IsConstant()) {
      values_[inst.result_id()] = inst.result_id();
    } else {
      values_[inst.result_id()] = kVaryingSSAId;
    }
  }

  original_id_bound_ = context()->module()->IdBound();
}

Pass::Status CCPPass::Process() {
  Initialize();

  // Process all entry point functions.
  ProcessFunction pfn = [this](Function* fp) { return PropagateConstants(fp); };
  bool modified = context()->ProcessReachableCallTree(pfn);
  return modified ? Pass::Status::SuccessWithChange
                  : Pass::Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
