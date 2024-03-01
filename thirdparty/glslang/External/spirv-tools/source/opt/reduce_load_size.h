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

#ifndef SOURCE_OPT_REDUCE_LOAD_SIZE_H_
#define SOURCE_OPT_REDUCE_LOAD_SIZE_H_

#include <unordered_map>

#include "source/opt/ir_context.h"
#include "source/opt/module.h"
#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class ReduceLoadSize : public Pass {
 public:
  explicit ReduceLoadSize(double replacement_threshold)
      : replacement_threshold_(replacement_threshold) {}

  const char* name() const override { return "reduce-load-size"; }
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
  // Replaces |inst|, which must be an OpCompositeExtract instruction, with
  // an OpAccessChain and a load if possible.  This happens only if it is a load
  // feeding |inst|.  Returns true if the substitution happened.  The position
  // of the new instructions will be in the same place as the load feeding the
  // extract.
  bool ReplaceExtract(Instruction* inst);

  // Returns true if the OpCompositeExtract instruction |inst| should be replace
  // or not.  This is determined by looking at the load that feeds |inst| if
  // it is a load.  |should_replace_cache_| is used to cache the results based
  // on the load feeding |inst|.
  bool ShouldReplaceExtract(Instruction* inst);

  // Threshold to determine whether we have to replace the load or not. If the
  // ratio of the used components of the load is less than the threshold, we
  // replace the load.
  double replacement_threshold_;

  // Maps the result id of an OpLoad instruction to the result of whether or
  // not the OpCompositeExtract that use the id should be replaced.
  std::unordered_map<uint32_t, bool> should_replace_cache_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_REDUCE_LOAD_SIZE_H_
