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

#ifndef SOURCE_OPT_PRIVATE_TO_LOCAL_PASS_H_
#define SOURCE_OPT_PRIVATE_TO_LOCAL_PASS_H_

#include "source/opt/ir_context.h"
#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// This pass implements total redundancy elimination.  This is the same as
// local redundancy elimination except it looks across basic block boundaries.
// An instruction, inst, is totally redundant if there is another instruction
// that dominates inst, and also computes the same value.
class PrivateToLocalPass : public Pass {
 public:
  const char* name() const override { return "private-to-local"; }
  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDefUse |
           IRContext::kAnalysisInstrToBlockMapping |
           IRContext::kAnalysisDecorations | IRContext::kAnalysisCombinators |
           IRContext::kAnalysisCFG | IRContext::kAnalysisDominatorAnalysis |
           IRContext::kAnalysisNameMap | IRContext::kAnalysisConstants |
           IRContext::kAnalysisTypes;
  }

 private:
  // Moves |variable| from the private storage class to the function storage
  // class of |function|.
  void MoveVariable(Instruction* variable, Function* function);

  // |inst| is an instruction declaring a varible.  If that variable is
  // referenced in a single function and all of uses are valid as defined by
  // |IsValidUse|, then that function is returned.  Otherwise, the return
  // value is |nullptr|.
  Function* FindLocalFunction(const Instruction& inst) const;

  // Returns true is |inst| is a valid use of a pointer.  In this case, a
  // valid use is one where the transformation is able to rewrite the type to
  // match a change in storage class of the original variable.
  bool IsValidUse(const Instruction* inst) const;

  // Given the result id of a pointer type, |old_type_id|, this function
  // returns the id of a the same pointer type except the storage class has
  // been changed to function.  If the type does not already exist, it will be
  // created.
  uint32_t GetNewType(uint32_t old_type_id);

  // Updates |inst|, and any instruction dependent on |inst|, to reflect the
  // change of the base pointer now pointing to the function storage class.
  void UpdateUse(Instruction* inst);
  void UpdateUses(uint32_t id);
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_PRIVATE_TO_LOCAL_PASS_H_
