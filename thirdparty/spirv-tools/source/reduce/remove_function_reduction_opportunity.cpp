// Copyright (c) 2019 Google LLC
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

#include "source/reduce/remove_function_reduction_opportunity.h"

#include "source/opt/eliminate_dead_functions_util.h"

namespace spvtools {
namespace reduce {

bool RemoveFunctionReductionOpportunity::PreconditionHolds() {
  // Removing one function cannot influence whether another function can be
  // removed.
  return true;
}

void RemoveFunctionReductionOpportunity::Apply() {
  for (opt::Module::iterator function_it = context_->module()->begin();
       function_it != context_->module()->end(); ++function_it) {
    if (&*function_it == function_) {
      function_it.Erase();
      context_->InvalidateAnalysesExceptFor(opt::IRContext::kAnalysisNone);
      return;
    }
  }
  assert(0 && "Function to be removed was not found.");
}

}  // namespace reduce
}  // namespace spvtools
