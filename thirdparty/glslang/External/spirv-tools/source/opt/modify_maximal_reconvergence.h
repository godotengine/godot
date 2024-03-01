// Copyright (c) 2024 Google LLC
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

#ifndef LIBSPIRV_OPT_MODIFY_MAXIMAL_RECONVERGENCE_H_
#define LIBSPIRV_OPT_MODIFY_MAXIMAL_RECONVERGENCE_H_

#include "pass.h"

namespace spvtools {
namespace opt {

// Modifies entry points to either add or remove MaximallyReconvergesKHR
//
// This pass will either add or remove MaximallyReconvergesKHR to all entry
// points in the module. When adding the execution mode, it does not attempt to
// determine whether any ray tracing invocation repack instructions might be
// executed because it is a runtime restriction. That is left to the user.
class ModifyMaximalReconvergence : public Pass {
 public:
  const char* name() const override { return "modify-maximal-reconvergence"; }
  Status Process() override;

  explicit ModifyMaximalReconvergence(bool add = true) : Pass(), add_(add) {}

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDefUse |
           IRContext::kAnalysisInstrToBlockMapping |
           IRContext::kAnalysisDecorations | IRContext::kAnalysisCombinators |
           IRContext::kAnalysisCFG | IRContext::kAnalysisNameMap |
           IRContext::kAnalysisConstants | IRContext::kAnalysisTypes;
  }

 private:
  bool AddMaximalReconvergence();
  bool RemoveMaximalReconvergence();

  bool add_;
};
}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_MODIFY_MAXIMAL_RECONVERGENCE_H_
