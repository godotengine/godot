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

#ifndef SOURCE_OPT_LOOP_FISSION_H_
#define SOURCE_OPT_LOOP_FISSION_H_

#include <algorithm>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "source/opt/cfg.h"
#include "source/opt/loop_dependence.h"
#include "source/opt/loop_utils.h"
#include "source/opt/module.h"
#include "source/opt/pass.h"
#include "source/opt/tree_iterator.h"

namespace spvtools {
namespace opt {

class LoopFissionPass : public Pass {
 public:
  // Function used to determine if a given loop should be split. Takes register
  // pressure region for that loop as a parameter and returns true if the loop
  // should be split.
  using FissionCriteriaFunction =
      std::function<bool(const RegisterLiveness::RegionRegisterLiveness&)>;

  // Pass built with this constructor will split all loops regardless of
  // register pressure. Will not split loops more than once.
  LoopFissionPass();

  // Split the loop if the number of registers used in the loop exceeds
  // |register_threshold_to_split|. |split_multiple_times| flag determines
  // whether or not the pass should split loops after already splitting them
  // once.
  LoopFissionPass(size_t register_threshold_to_split,
                  bool split_multiple_times = true);

  // Split loops whose register pressure meets the criteria of |functor|.
  LoopFissionPass(FissionCriteriaFunction functor,
                  bool split_multiple_times = true)
      : split_criteria_(functor), split_multiple_times_(split_multiple_times) {}

  const char* name() const override { return "loop-fission"; }

  Pass::Status Process() override;

  // Checks if |loop| meets the register pressure criteria to be split.
  bool ShouldSplitLoop(const Loop& loop, IRContext* context);

 private:
  // Functor to run in ShouldSplitLoop to determine if the register pressure
  // criteria is met for splitting the loop.
  FissionCriteriaFunction split_criteria_;

  // Flag designating whether or not we should also split the result of
  // previously split loops if they meet the register presure criteria.
  bool split_multiple_times_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_LOOP_FISSION_H_
