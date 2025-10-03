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

#ifndef SOURCE_OPT_LOOP_UNROLLER_H_
#define SOURCE_OPT_LOOP_UNROLLER_H_

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

class LoopUnroller : public Pass {
 public:
  LoopUnroller() : Pass(), fully_unroll_(true), unroll_factor_(0) {}
  LoopUnroller(bool fully_unroll, int unroll_factor)
      : Pass(), fully_unroll_(fully_unroll), unroll_factor_(unroll_factor) {}

  const char* name() const override { return "loop-unroll"; }

  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDefUse |
           IRContext::kAnalysisInstrToBlockMapping |
           IRContext::kAnalysisDecorations | IRContext::kAnalysisCombinators |
           IRContext::kAnalysisNameMap | IRContext::kAnalysisConstants |
           IRContext::kAnalysisTypes;
  }

 private:
  bool fully_unroll_;
  int unroll_factor_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_LOOP_UNROLLER_H_
