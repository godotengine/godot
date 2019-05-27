// Copyright (c) 2019 Google Inc.
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

#ifndef SOURCE_OPT_GENERATE_WEBGPU_INITIALIZERS_PASS_H_
#define SOURCE_OPT_GENERATE_WEBGPU_INITIALIZERS_PASS_H_

#include "source/opt/ir_context.h"
#include "source/opt/module.h"
#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// Adds initializers to variables with storage classes Output, Private, and
// Function if they are missing. In the WebGPU environment these storage classes
// require that the variables are initialized. Currently they are initialized to
// NULL, though in the future some of them may be initialized to the first value
// that is stored in them, if that was a constant.
class GenerateWebGPUInitializersPass : public Pass {
 public:
  const char* name() const override { return "generate-webgpu-initializers"; }
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

 private:
  using NullConstantTypeMap = std::unordered_map<uint32_t, Instruction*>;
  NullConstantTypeMap null_constant_type_map_;
  std::unordered_set<Instruction*> seen_null_constants_;

  Instruction* GetNullConstantForVariable(Instruction* variable_inst);
  void AddNullInitializerToVariable(Instruction* constant_inst,
                                    Instruction* variable_inst);
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_GENERATE_WEBGPU_INITIALIZERS_PASS_H_
