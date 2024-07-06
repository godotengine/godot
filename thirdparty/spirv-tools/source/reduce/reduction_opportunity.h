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

#ifndef SOURCE_REDUCE_REDUCTION_OPPORTUNITY_H_
#define SOURCE_REDUCE_REDUCTION_OPPORTUNITY_H_

#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace reduce {

// Abstract class: an opportunity to apply a reducing transformation.
class ReductionOpportunity {
 public:
  ReductionOpportunity() = default;
  virtual ~ReductionOpportunity() = default;

  // Returns true if this opportunity has not been disabled by the application
  // of another conflicting opportunity.
  virtual bool PreconditionHolds() = 0;

  // Applies the opportunity, mutating the module from which the opportunity was
  // created. It is a no-op if PreconditionHolds() returns false.
  void TryToApply();

 protected:
  // Applies the opportunity, mutating the module from which the opportunity was
  // created.
  // Precondition: PreconditionHolds() must return true.
  virtual void Apply() = 0;
};

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_REDUCTION_OPPORTUNITY_H_
