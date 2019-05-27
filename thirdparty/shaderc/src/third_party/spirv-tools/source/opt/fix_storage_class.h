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
  bool PropagateStorageClass(Instruction* inst, SpvStorageClass storage_class);

  // Changes the storage class of the result of |inst| to |storage_class|.
  // Is it assumed that the result type of |inst| is a pointer type.
  // Propagates the change to the users of |inst| as well.
  // Returns true of any changes were made.
  void FixInstruction(Instruction* inst, SpvStorageClass storage_class);

  // Changes the storage class of the result of |inst| to |storage_class|.  The
  // result type of |inst| must be a pointer.
  void ChangeResultStorageClass(Instruction* inst,
                                SpvStorageClass storage_class) const;

  // Returns true if the result type of |inst| is a pointer.
  bool IsPointerResultType(Instruction* inst);

  // Returns true if the result of |inst| is a pointer to storage class
  // |storage_class|.
  bool IsPointerToStorageClass(Instruction* inst,
                               SpvStorageClass storage_class);
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_FIX_STORAGE_CLASS_H_
