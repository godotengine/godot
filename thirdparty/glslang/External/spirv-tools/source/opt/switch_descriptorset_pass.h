// Copyright (c) 2023 LunarG Inc.
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

#pragma once

#include <cstdio>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class SwitchDescriptorSetPass : public Pass {
 public:
  SwitchDescriptorSetPass(uint32_t ds_from, uint32_t ds_to)
      : ds_from_(ds_from), ds_to_(ds_to) {}

  const char* name() const override { return "switch-descriptorset"; }

  Status Process() override;

  IRContext::Analysis GetPreservedAnalyses() override {
    // this pass preserves everything except decorations
    uint32_t mask = ((IRContext::kAnalysisEnd << 1) - 1);
    mask &= ~static_cast<uint32_t>(IRContext::kAnalysisDecorations);
    return static_cast<IRContext::Analysis>(mask);
  }

 private:
  uint32_t ds_from_;
  uint32_t ds_to_;
};

}  // namespace opt
}  // namespace spvtools
