// Copyright (c) 2018 Google LLC.
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

#ifndef SOURCE_OPT_VECTOR_DCE_H_
#define SOURCE_OPT_VECTOR_DCE_H_

#include <unordered_map>
#include <vector>

#include "source/opt/mem_pass.h"
#include "source/util/bit_vector.h"

namespace spvtools {
namespace opt {

class VectorDCE : public MemPass {
 private:
  using LiveComponentMap = std::unordered_map<uint32_t, utils::BitVector>;

  // According to the SPEC the maximum size for a vector is 16.  See the data
  // rules in the universal validation rules (section 2.16.1).
  enum { kMaxVectorSize = 16 };

  struct WorkListItem {
    WorkListItem() : instruction(nullptr), components(kMaxVectorSize) {}

    Instruction* instruction;
    utils::BitVector components;
  };

 public:
  VectorDCE() : all_components_live_(kMaxVectorSize) {
    for (uint32_t i = 0; i < kMaxVectorSize; i++) {
      all_components_live_.Set(i);
    }
  }

  const char* name() const override { return "vector-dce"; }
  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDefUse | IRContext::kAnalysisCFG |
           IRContext::kAnalysisInstrToBlockMapping |
           IRContext::kAnalysisLoopAnalysis | IRContext::kAnalysisDecorations |
           IRContext::kAnalysisDominatorAnalysis | IRContext::kAnalysisNameMap |
           IRContext::kAnalysisConstants | IRContext::kAnalysisTypes;
  }

 private:
  // Runs the vector dce pass on |function|.  Returns true if |function| was
  // modified.
  bool VectorDCEFunction(Function* function);

  // Identifies the live components of the vectors that are results of
  // instructions in |function|.  The results are stored in |live_components|.
  void FindLiveComponents(Function* function,
                          LiveComponentMap* live_components);

  // Rewrites instructions in |function| that are dead or partially dead.  If an
  // instruction does not have an entry in |live_components|, then it is not
  // changed.  Returns true if |function| was modified.
  bool RewriteInstructions(Function* function,
                           const LiveComponentMap& live_components);

  // Rewrites the OpCompositeInsert instruction |current_inst| to avoid
  // unnecessary computes given that the only components of the result that are
  // live are |live_components|.
  //
  // If the value being inserted is not live, then the result of |current_inst|
  // is replaced by the composite input to |current_inst|.
  //
  // If the composite input to |current_inst| is not live, then it is replaced
  // by and OpUndef in |current_inst|.
  bool RewriteInsertInstruction(Instruction* current_inst,
                                const utils::BitVector& live_components);

  // Returns true if the result of |inst| is a vector or a scalar.
  bool HasVectorOrScalarResult(const Instruction* inst) const;

  // Returns true if the result of |inst| is a scalar.
  bool HasVectorResult(const Instruction* inst) const;

  // Returns true if the result of |inst| is a vector.
  bool HasScalarResult(const Instruction* inst) const;

  // Adds |work_item| to |work_list| if it is not already live according to
  // |live_components|.  |live_components| is updated to indicate that
  // |work_item| is now live.
  void AddItemToWorkListIfNeeded(WorkListItem work_item,
                                 LiveComponentMap* live_components,
                                 std::vector<WorkListItem>* work_list);

  // Marks the components |live_elements| of the uses in |current_inst| as live
  // according to |live_components|. If they were not live before, then they are
  // added to |work_list|.
  void MarkUsesAsLive(Instruction* current_inst,
                      const utils::BitVector& live_elements,
                      LiveComponentMap* live_components,
                      std::vector<WorkListItem>* work_list);

  // Marks the uses in the OpVectorShuffle instruction in |current_item| as live
  // based on the live components in |current_item|. If anything becomes live
  // they are added to |work_list| and |live_components| is updated
  // accordingly.
  void MarkVectorShuffleUsesAsLive(const WorkListItem& current_item,
                                   VectorDCE::LiveComponentMap* live_components,
                                   std::vector<WorkListItem>* work_list);

  // Marks the uses in the OpCompositeInsert instruction in |current_item| as
  // live based on the live components in |current_item|. If anything becomes
  // live they are added to |work_list| and |live_components| is updated
  // accordingly.
  void MarkInsertUsesAsLive(const WorkListItem& current_item,
                            LiveComponentMap* live_components,
                            std::vector<WorkListItem>* work_list);

  // Marks the uses in the OpCompositeExtract instruction |current_inst| as
  // live. If anything becomes live they are added to |work_list| and
  // |live_components| is updated accordingly.
  void MarkExtractUseAsLive(const Instruction* current_inst,
                            const utils::BitVector& live_elements,
                            LiveComponentMap* live_components,
                            std::vector<WorkListItem>* work_list);

  // Marks the uses in the OpCompositeConstruct instruction |current_inst| as
  // live. If anything becomes live they are added to |work_list| and
  // |live_components| is updated accordingly.
  void MarkCompositeContructUsesAsLive(WorkListItem work_item,
                                       LiveComponentMap* live_components,
                                       std::vector<WorkListItem>* work_list);

  // A BitVector that can always be used to say that all components of a vector
  // are live.
  utils::BitVector all_components_live_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_VECTOR_DCE_H_
