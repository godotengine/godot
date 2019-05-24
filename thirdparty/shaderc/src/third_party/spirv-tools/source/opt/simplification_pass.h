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

#ifndef SOURCE_OPT_SIMPLIFICATION_PASS_H_
#define SOURCE_OPT_SIMPLIFICATION_PASS_H_

#include "source/opt/function.h"
#include "source/opt/ir_context.h"
#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class SimplificationPass : public Pass {
 public:
  const char* name() const override { return "simplify-instructions"; }
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
  // Returns true if the module was changed.  The simplifier is called on every
  // instruction in |function| until nothing else in the function can be
  // simplified.
  bool SimplifyFunction(Function* function);
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_SIMPLIFICATION_PASS_H_
