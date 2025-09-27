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

#ifndef SOURCE_OPT_ELIMINATE_DEAD_OUTPUT_STORES_H_
#define SOURCE_OPT_ELIMINATE_DEAD_OUTPUT_STORES_H_

#include <unordered_set>

#include "source/opt/ir_context.h"
#include "source/opt/module.h"
#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class EliminateDeadOutputStoresPass : public Pass {
 public:
  explicit EliminateDeadOutputStoresPass(
      std::unordered_set<uint32_t>* live_locs,
      std::unordered_set<uint32_t>* live_builtins)
      : live_locs_(live_locs), live_builtins_(live_builtins) {}

  const char* name() const override { return "eliminate-dead-output-stores"; }
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
  // Initialize elimination
  void InitializeElimination();

  // Do dead output store analysis
  Status DoDeadOutputStoreAnalysis();

  // Do dead output store analysis
  Status DoDeadOutputStoreElimination();

  // Mark all locations live
  void MarkAllLocsLive();

  // Kill all stores resulting from |ref|.
  void KillAllStoresOfRef(Instruction* ref);

  // Kill all dead stores resulting from |user| of loc-based |var|.
  void KillAllDeadStoresOfLocRef(Instruction* user, Instruction* var);

  // Kill all dead stores resulting from |user| of builtin |var|.
  void KillAllDeadStoresOfBuiltinRef(Instruction* user, Instruction* var);

  // Return true if any of |count| locations starting at location |start| are
  // live.
  bool AnyLocsAreLive(uint32_t start, uint32_t count);

  // Return true if builtin |bi| is live.
  bool IsLiveBuiltin(uint32_t bi);

  std::unordered_set<uint32_t>* live_locs_;
  std::unordered_set<uint32_t>* live_builtins_;

  std::vector<Instruction*> kill_list_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_ELIMINATE_DEAD_OUTPUT_STORES_H_
