// Copyright (c) 2021 The Khronos Group Inc.
// Copyright (c) 2021 Valve Corporation
// Copyright (c) 2021 LunarG Inc.
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

#ifndef SOURCE_OPT_INTERP_FIXUP_H
#define SOURCE_OPT_INTERP_FIXUP_H

#include "source/opt/ir_context.h"
#include "source/opt/module.h"
#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// Replaces overloaded internal form for GLSLstd450Interpolate* instructions
// with external form. Specifically, removes OpLoad from the first argument
// and replaces it with the pointer for the OpLoad. glslang generates the
// internal form. This pass is called as part of glslang HLSL legalization.
class InterpFixupPass : public Pass {
 public:
  const char* name() const override { return "interp-fixup"; }
  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisInstrToBlockMapping |
           IRContext::kAnalysisDecorations | IRContext::kAnalysisCombinators |
           IRContext::kAnalysisCFG | IRContext::kAnalysisDominatorAnalysis |
           IRContext::kAnalysisLoopAnalysis | IRContext::kAnalysisNameMap |
           IRContext::kAnalysisScalarEvolution |
           IRContext::kAnalysisRegisterPressure |
           IRContext::kAnalysisValueNumberTable |
           IRContext::kAnalysisStructuredCFG |
           IRContext::kAnalysisBuiltinVarId |
           IRContext::kAnalysisIdToFuncMapping | IRContext::kAnalysisTypes |
           IRContext::kAnalysisDefUse | IRContext::kAnalysisConstants;
  }
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_INTERP_FIXUP_H
