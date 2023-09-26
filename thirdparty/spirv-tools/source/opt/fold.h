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

#ifndef SOURCE_OPT_FOLD_H_
#define SOURCE_OPT_FOLD_H_

#include <cstdint>
#include <vector>

#include "source/opt/const_folding_rules.h"
#include "source/opt/constants.h"
#include "source/opt/def_use_manager.h"
#include "source/opt/folding_rules.h"

namespace spvtools {
namespace opt {

class InstructionFolder {
 public:
  explicit InstructionFolder(IRContext* context)
      : context_(context),
        const_folding_rules_(new ConstantFoldingRules(context)),
        folding_rules_(new FoldingRules(context)) {
    folding_rules_->AddFoldingRules();
    const_folding_rules_->AddFoldingRules();
  }

  explicit InstructionFolder(
      IRContext* context, std::unique_ptr<FoldingRules>&& folding_rules,
      std::unique_ptr<ConstantFoldingRules>&& constant_folding_rules)
      : context_(context),
        const_folding_rules_(std::move(constant_folding_rules)),
        folding_rules_(std::move(folding_rules)) {
    folding_rules_->AddFoldingRules();
    const_folding_rules_->AddFoldingRules();
  }

  // Returns the result of folding a scalar instruction with the given |opcode|
  // and |operands|. Each entry in |operands| is a pointer to an
  // analysis::Constant instance, which should've been created with the constant
  // manager (See IRContext::get_constant_mgr).
  //
  // It is an error to call this function with an opcode that does not pass the
  // IsFoldableOpcode test. If any error occurs during folding, the folder will
  // fail with a call to assert.
  uint32_t FoldScalars(
      spv::Op opcode,
      const std::vector<const analysis::Constant*>& operands) const;

  // Returns the result of performing an operation with the given |opcode| over
  // constant vectors with |num_dims| dimensions.  Each entry in |operands| is a
  // pointer to an analysis::Constant instance, which should've been created
  // with the constant manager (See IRContext::get_constant_mgr).
  //
  // This function iterates through the given vector type constant operands and
  // calculates the result for each element of the result vector to return.
  // Vectors with longer than 32-bit scalar components are not accepted in this
  // function.
  //
  // It is an error to call this function with an opcode that does not pass the
  // IsFoldableOpcode test. If any error occurs during folding, the folder will
  // fail with a call to assert.
  std::vector<uint32_t> FoldVectors(
      spv::Op opcode, uint32_t num_dims,
      const std::vector<const analysis::Constant*>& operands) const;

  // Returns true if |opcode| represents an operation handled by FoldScalars or
  // FoldVectors.
  bool IsFoldableOpcode(spv::Op opcode) const;

  // Returns true if |cst| is supported by FoldScalars and FoldVectors.
  bool IsFoldableConstant(const analysis::Constant* cst) const;

  // Returns true if |FoldInstructionToConstant| could fold an instruction whose
  // result type is |type_inst|.
  bool IsFoldableType(Instruction* type_inst) const;

  // Tries to fold |inst| to a single constant, when the input ids to |inst|
  // have been substituted using |id_map|.  Returns a pointer to the OpConstant*
  // instruction if successful.  If necessary, a new constant instruction is
  // created and placed in the global values section.
  //
  // |id_map| is a function that takes one result id and returns another.  It
  // can be used for things like CCP where it is known that some ids contain a
  // constant, but the instruction itself has not been updated yet.  This can
  // map those ids to the appropriate constants.
  Instruction* FoldInstructionToConstant(
      Instruction* inst, std::function<uint32_t(uint32_t)> id_map) const;
  // Returns true if |inst| can be folded into a simpler instruction.
  // If |inst| can be simplified, |inst| is overwritten with the simplified
  // instruction reusing the same result id.
  //
  // If |inst| is simplified, it is possible that the resulting code in invalid
  // because the instruction is in a bad location.  Callers of this function
  // have to handle the following cases:
  //
  // 1) An OpPhi becomes and OpCopyObject - If there are OpPhi instruction after
  //    |inst| in a basic block then this is invalid.  The caller must fix this
  //    up.
  bool FoldInstruction(Instruction* inst) const;

  // Return true if this opcode has a const folding rule associtated with it.
  bool HasConstFoldingRule(const Instruction* inst) const {
    return GetConstantFoldingRules().HasFoldingRule(inst);
  }

 private:
  // Returns a reference to the ConstnatFoldingRules instance.
  const ConstantFoldingRules& GetConstantFoldingRules() const {
    return *const_folding_rules_;
  }

  // Returns a reference to the FoldingRules instance.
  const FoldingRules& GetFoldingRules() const { return *folding_rules_; }

  // Returns the single-word result from performing the given unary operation on
  // the operand value which is passed in as a 32-bit word.
  uint32_t UnaryOperate(spv::Op opcode, uint32_t operand) const;

  // Returns the single-word result from performing the given binary operation
  // on the operand values which are passed in as two 32-bit word.
  uint32_t BinaryOperate(spv::Op opcode, uint32_t a, uint32_t b) const;

  // Returns the single-word result from performing the given ternary operation
  // on the operand values which are passed in as three 32-bit word.
  uint32_t TernaryOperate(spv::Op opcode, uint32_t a, uint32_t b,
                          uint32_t c) const;

  // Returns the single-word result from performing the given operation on the
  // operand words. This only works with 32-bit operations and uses boolean
  // convention that 0u is false, and anything else is boolean true.
  // TODO(qining): Support operands other than 32-bit wide.
  uint32_t OperateWords(spv::Op opcode,
                        const std::vector<uint32_t>& operand_words) const;

  bool FoldInstructionInternal(Instruction* inst) const;

  // Returns true if |inst| is a binary operation that takes two integers as
  // parameters and folds to a constant that can be represented as an unsigned
  // 32-bit value when the ids have been replaced by |id_map|.  If |inst| can be
  // folded, the resulting value is returned in |*result|.  Valid result types
  // for the instruction are any integer (signed or unsigned) with 32-bits or
  // less, or a boolean value.
  bool FoldBinaryIntegerOpToConstant(
      Instruction* inst, const std::function<uint32_t(uint32_t)>& id_map,
      uint32_t* result) const;

  // Returns true if |inst| is a binary operation on two boolean values, and
  // folds
  // to a constant boolean value when the ids have been replaced using |id_map|.
  // If |inst| can be folded, the result value is returned in |*result|.
  bool FoldBinaryBooleanOpToConstant(
      Instruction* inst, const std::function<uint32_t(uint32_t)>& id_map,
      uint32_t* result) const;

  // Returns true if |inst| can be folded to an constant when the ids have been
  // substituted using id_map.  If it can, the value is returned in |result|. If
  // not, |result| is unchanged.  It is assumed that not all operands are
  // constant.  Those cases are handled by |FoldScalar|.
  bool FoldIntegerOpToConstant(Instruction* inst,
                               const std::function<uint32_t(uint32_t)>& id_map,
                               uint32_t* result) const;

  IRContext* context_;

  // Folding rules used by |FoldInstructionToConstant| and |FoldInstruction|.
  std::unique_ptr<ConstantFoldingRules> const_folding_rules_;

  // Folding rules used by |FoldInstruction|.
  std::unique_ptr<FoldingRules> folding_rules_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_FOLD_H_
