// Copyright (c) 2021 Alastair F. Donaldson
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

#ifndef SOURCE_REDUCE_STRUCTURED_CONSTRUCT_TO_BLOCK_REDUCTION_OPPORTUNITY_H_
#define SOURCE_REDUCE_STRUCTURED_CONSTRUCT_TO_BLOCK_REDUCTION_OPPORTUNITY_H_

#include "source/opt/ir_context.h"
#include "source/reduce/reduction_opportunity.h"

namespace spvtools {
namespace reduce {

// An opportunity to replace a skeletal structured control flow construct with a
// single block.
class StructuredConstructToBlockReductionOpportunity
    : public ReductionOpportunity {
 public:
  // Constructs an opportunity from a header block id.
  StructuredConstructToBlockReductionOpportunity(opt::IRContext* context,
                                                 uint32_t construct_header)
      : context_(context), construct_header_(construct_header) {}

  // Returns true if and only if |construct_header_| exists in the module -
  // another opportunity may have removed it.
  bool PreconditionHolds() override;

 protected:
  void Apply() override;

 private:
  opt::IRContext* context_;
  uint32_t construct_header_;
};

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_STRUCTURED_CONSTRUCT_TO_BLOCK_REDUCTION_OPPORTUNITY_H_
