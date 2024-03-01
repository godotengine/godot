// Copyright (c) 2018 The Khronos Group Inc.
// Copyright (c) 2018 Valve Corporation
// Copyright (c) 2018 LunarG Inc.
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

#ifndef SOURCE_OPT_DEAD_INSERT_ELIM_PASS_H_
#define SOURCE_OPT_DEAD_INSERT_ELIM_PASS_H_

#include <algorithm>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "source/opt/basic_block.h"
#include "source/opt/def_use_manager.h"
#include "source/opt/ir_context.h"
#include "source/opt/mem_pass.h"
#include "source/opt/module.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class DeadInsertElimPass : public MemPass {
 public:
  DeadInsertElimPass() = default;

  const char* name() const override { return "eliminate-dead-inserts"; }
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
  // Return the number of subcomponents in the composite type |typeId|.
  // Return 0 if not a composite type or number of components is not a
  // 32-bit constant.
  uint32_t NumComponents(Instruction* typeInst);

  // Mark all inserts in instruction chain ending at |insertChain| with
  // indices that intersect with extract indices |extIndices| starting with
  // index at |extOffset|. Chains are composed solely of Inserts and Phis.
  // Mark all inserts in chain if |extIndices| is nullptr.
  void MarkInsertChain(Instruction* insertChain,
                       std::vector<uint32_t>* extIndices, uint32_t extOffset,
                       std::unordered_set<uint32_t>* visited_phis);

  // Perform EliminateDeadInsertsOnePass(|func|) until no modification is
  // made. Return true if modified.
  bool EliminateDeadInserts(Function* func);

  // DCE all dead struct, matrix and vector inserts in |func|. An insert is
  // dead if the value it inserts is never used. Replace any reference to the
  // insert with its original composite. Return true if modified. Dead inserts
  // in dependence cycles are not currently eliminated. Dead inserts into
  // arrays are not currently eliminated.
  bool EliminateDeadInsertsOnePass(Function* func);

  // Return true if all extensions in this module are allowed by this pass.
  bool AllExtensionsSupported() const;

  // Live inserts
  std::unordered_set<uint32_t> liveInserts_;

  // Visited phis as insert chain is traversed; used to avoid infinite loop
  std::unordered_map<uint32_t, bool> visitedPhis_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_DEAD_INSERT_ELIM_PASS_H_
