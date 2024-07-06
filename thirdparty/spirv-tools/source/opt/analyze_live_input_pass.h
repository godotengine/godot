// Copyright (c) 2022 The Khronos Group Inc.
// Copyright (c) 2022 LunarG Inc.
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

#ifndef SOURCE_OPT_ANALYZE_LIVE_INPUT_H_
#define SOURCE_OPT_ANALYZE_LIVE_INPUT_H_

#include <unordered_set>

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class AnalyzeLiveInputPass : public Pass {
 public:
  explicit AnalyzeLiveInputPass(std::unordered_set<uint32_t>* live_locs,
                                std::unordered_set<uint32_t>* live_builtins)
      : live_locs_(live_locs), live_builtins_(live_builtins) {}

  const char* name() const override { return "analyze-live-input"; }
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
  // Do live input analysis
  Status DoLiveInputAnalysis();

  std::unordered_set<uint32_t>* live_locs_;
  std::unordered_set<uint32_t>* live_builtins_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_ANALYZE_LIVE_INPUT_H_
