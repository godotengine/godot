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

#ifndef SOURCE_OPT_DESC_SROA_H_
#define SOURCE_OPT_DESC_SROA_H_

#include <cstdio>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "source/opt/function.h"
#include "source/opt/pass.h"
#include "source/opt/type_manager.h"

namespace spvtools {
namespace opt {

// Documented in optimizer.hpp
class DescriptorScalarReplacement : public Pass {
 public:
  DescriptorScalarReplacement() {}

  const char* name() const override { return "descriptor-scalar-replacement"; }

  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDefUse |
           IRContext::kAnalysisInstrToBlockMapping |
           IRContext::kAnalysisCombinators | IRContext::kAnalysisCFG |
           IRContext::kAnalysisConstants | IRContext::kAnalysisTypes;
  }

 private:
  // Replaces all references to |var| by new variables, one for each element of
  // the array |var|.  The binding for the new variables corresponding to
  // element i will be the binding of |var| plus i.  Returns true if successful.
  bool ReplaceCandidate(Instruction* var);

  // Replaces the base address |var| in the OpAccessChain or
  // OpInBoundsAccessChain instruction |use| by the variable that the access
  // chain accesses.  The first index in |use| must be an |OpConstant|.  Returns
  // |true| if successful.
  bool ReplaceAccessChain(Instruction* var, Instruction* use);

  // Replaces the given compososite variable |var| loaded by OpLoad |value| with
  // replacement variables, one for each component that's accessed in the
  // shader. Assumes that |value| is only used by OpCompositeExtract
  // instructions, one index at a time. Returns true on success, and false
  // otherwise.
  bool ReplaceLoadedValue(Instruction* var, Instruction* value);

  // Replaces the given OpCompositeExtract |extract| and all of its references
  // with an OpLoad of a replacement variable. |var| is the variable with
  // composite type whose value is being used by |extract|. Assumes that
  // |extract| is extracting one index only. Returns true on success, and false
  // otherwise.
  bool ReplaceCompositeExtract(Instruction* var, Instruction* extract);

  // Returns the id of the variable that will be used to replace the |idx|th
  // element of |var|.  The variable is created if it has not already been
  // created.
  uint32_t GetReplacementVariable(Instruction* var, uint32_t idx);

  // Returns the id of a new variable that can be used to replace the |idx|th
  // element of |var|.
  uint32_t CreateReplacementVariable(Instruction* var, uint32_t idx);

  // Returns the number of bindings used by the given |type_id|.
  // All types are considered to use 1 binding slot, except:
  // 1- A pointer type consumes as many binding numbers as its pointee.
  // 2- An array of size N consumes N*M binding numbers, where M is the number
  // of bindings used by each array element.
  // 3- The number of bindings consumed by a structure is the sum of the
  // bindings used by its members.
  uint32_t GetNumBindingsUsedByType(uint32_t type_id);

  // Copy all of the decorations of variable |old_var| and make them as
  // decorations for the new variable whose id is |new_var_id|. The new variable
  // is supposed to replace |index|th element of |old_var|.
  // |new_var_ptr_type_id| is the id of the pointer to the type of the new
  // variable. |is_old_var_array| is true if |old_var| has an array type.
  // |is_old_var_struct| is true if |old_var| has a structure type.
  // |old_var_type| is the pointee type of |old_var|.
  void CopyDecorationsForNewVariable(Instruction* old_var, uint32_t index,
                                     uint32_t new_var_id,
                                     uint32_t new_var_ptr_type_id,
                                     const bool is_old_var_array,
                                     const bool is_old_var_struct,
                                     Instruction* old_var_type);

  // Get the new binding number for a new variable that will be replaced with an
  // |index|th element of an old variable. The old variable has |old_binding|
  // as its binding number. |ptr_elem_type_id| the id of the pointer to the
  // element type. |is_old_var_array| is true if the old variable has an array
  // type. |is_old_var_struct| is true if the old variable has a structure type.
  // |old_var_type| is the pointee type of the old variable.
  uint32_t GetNewBindingForElement(uint32_t old_binding, uint32_t index,
                                   uint32_t ptr_elem_type_id,
                                   const bool is_old_var_array,
                                   const bool is_old_var_struct,
                                   Instruction* old_var_type);

  // Create a new OpDecorate(String) instruction by cloning |old_decoration|.
  // The new OpDecorate(String) instruction will be used for a variable whose id
  // is |new_var_ptr_type_id|. If |old_decoration| is a decoration for a
  // binding, the new OpDecorate(String) instruction will have |new_binding| as
  // its binding.
  void CreateNewDecorationForNewVariable(Instruction* old_decoration,
                                         uint32_t new_var_id,
                                         uint32_t new_binding);

  // Create a new OpDecorate instruction whose operand is the same as an
  // OpMemberDecorate instruction |old_member_decoration| except Target operand.
  // The Target operand of the new OpDecorate instruction will be |new_var_id|.
  void CreateNewDecorationForMemberDecorate(Instruction* old_decoration,
                                            uint32_t new_var_id);

  // A map from an OpVariable instruction to the set of variables that will be
  // used to replace it. The entry |replacement_variables_[var][i]| is the id of
  // a variable that will be used in the place of the ith element of the
  // array |var|. If the entry is |0|, then the variable has not been
  // created yet.
  std::map<Instruction*, std::vector<uint32_t>> replacement_variables_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_DESC_SROA_H_
