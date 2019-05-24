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

#include "source/reduce/reduction_pass.h"

#include <algorithm>

#include "source/opt/build_module.h"

namespace spvtools {
namespace reduce {

std::vector<uint32_t> ReductionPass::TryApplyReduction(
    const std::vector<uint32_t>& binary) {
  // We represent modules as binaries because (a) attempts at reduction need to
  // end up in binary form to be passed on to SPIR-V-consuming tools, and (b)
  // when we apply a reduction step we need to do it on a fresh version of the
  // module as if the reduction step proves to be uninteresting we need to
  // backtrack; re-parsing from binary provides a very clean way of cloning the
  // module.
  std::unique_ptr<opt::IRContext> context =
      BuildModule(target_env_, consumer_, binary.data(), binary.size());
  assert(context);

  std::vector<std::unique_ptr<ReductionOpportunity>> opportunities =
      finder_->GetAvailableOpportunities(context.get());

  // There is no point in having a granularity larger than the number of
  // opportunities, so reduce the granularity in this case.
  if (granularity_ > opportunities.size()) {
    granularity_ = std::max((uint32_t)1, (uint32_t)opportunities.size());
  }

  assert(granularity_ > 0);

  if (index_ >= opportunities.size()) {
    // We have reached the end of the available opportunities and, therefore,
    // the end of the round for this pass, so reset the index and decrease the
    // granularity for the next round. Return an empty vector to signal the end
    // of the round.
    index_ = 0;
    granularity_ = std::max((uint32_t)1, granularity_ / 2);
    return std::vector<uint32_t>();
  }

  for (uint32_t i = index_;
       i < std::min(index_ + granularity_, (uint32_t)opportunities.size());
       ++i) {
    opportunities[i]->TryToApply();
  }

  std::vector<uint32_t> result;
  context->module()->ToBinary(&result, false);
  return result;
}

void ReductionPass::SetMessageConsumer(MessageConsumer consumer) {
  consumer_ = std::move(consumer);
}

bool ReductionPass::ReachedMinimumGranularity() const {
  assert(granularity_ != 0);
  return granularity_ == 1;
}

std::string ReductionPass::GetName() const { return finder_->GetName(); }

void ReductionPass::NotifyInteresting(bool interesting) {
  if (!interesting) {
    index_ += granularity_;
  }
}

}  // namespace reduce
}  // namespace spvtools
