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

#ifndef SOURCE_OPT_FIX_STORAGE_CLASS_H_
#define SOURCE_OPT_FIX_STORAGE_CLASS_H_

#include <unordered_map>

#include "source/opt/ir_context.h"
#include "source/opt/module.h"
#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// This pass tries to fix validation error due to a mismatch of storage classes
// in instructions.  There is no guarantee that all such error will be fixed,
// and it is possible that in fixing these errors, it could lead to other
// errors.
class FixStorageClass : public Pass {
 public:
  const char* name() const override { return "fix-storage-class"; }
  Status Process() override;

  // Return the mask of preserved Analyses.
  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDefUse |
           IRContext::kAnalysisInstrToBlockMapping |
           IRContext::kAnalysisCombinators | IRContext::kAnalysisCFG |
           IRContext::kAnalysisDominatorAnalysis |
           IRContext::kAnalysisLoopAnalysis | IRContext::kAnalysisNameMap |
           IRContext::kAnalysisConstants | IRContext::kAnalysisTypes;
  }

 private:
  // Changes the storage class of the result of |inst| to |storage_class| in
  // appropriate, and propagates the change to the users of |inst| as well.
  // Returns true of any changes were made.
  // |seen| is used to track OpPhi instructions that should not be processed.
  bool PropagateStorageClass(Instruction* inst, spv::StorageClass storage_class,
                             std::set<uint32_t>* seen);

  // Changes the storage class of the result of |inst| to |storage_class|.
  // Is it assumed that the result type of |inst| is a pointer type.
  // Propagates the change to the users of |inst| as well.
  // Returns true of any changes were made.
  // |seen| is used to track OpPhi instructions that should not be processed by
  // |PropagateStorageClass|
  void FixInstructionStorageClass(Instruction* inst,
                                  spv::StorageClass storage_class,
                                  std::set<uint32_t>* seen);

  // Changes the storage class of the result of |inst| to |storage_class|.  The
  // result type of |inst| must be a pointer.
  void ChangeResultStorageClass(Instruction* inst,
                                spv::StorageClass storage_class) const;

  // Returns true if the result type of |inst| is a pointer.
  bool IsPointerResultType(Instruction* inst);

  // Returns true if the result of |inst| is a pointer to storage class
  // |storage_class|.
  bool IsPointerToStorageClass(Instruction* inst,
                               spv::StorageClass storage_class);

  // Change |inst| to match that operand |op_idx| now has type |type_id|, and
  // adjust any uses of |inst| accordingly. Returns true if the code changed.
  bool PropagateType(Instruction* inst, uint32_t type_id, uint32_t op_idx,
                     std::set<uint32_t>* seen);

  // Changes the result type of |inst| to |new_type_id|.
  bool ChangeResultType(Instruction* inst, uint32_t new_type_id);

  // Returns the type id of the member of the type |id| that would be returned
  // by following the indices of the access chain instruction |inst|.
  uint32_t WalkAccessChainType(Instruction* inst, uint32_t id);
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_FIX_STORAGE_CLASS_H_
