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

#ifndef SOURCE_REDUCE_REDUCTION_UTIL_H_
#define SOURCE_REDUCE_REDUCTION_UTIL_H_

#include "spirv-tools/libspirv.hpp"

#include "source/opt/ir_context.h"
#include "source/reduce/reduction_opportunity.h"

namespace spvtools {
namespace reduce {

extern const uint32_t kTrueBranchOperandIndex;
extern const uint32_t kFalseBranchOperandIndex;

// Returns a global OpVariable of type |pointer_type_id|, adding one if none
// exist.
uint32_t FindOrCreateGlobalVariable(opt::IRContext* context,
                                    uint32_t pointer_type_id);

// Returns an OpVariable of type |pointer_type_id| declared in |function|,
// adding one if none exist.
uint32_t FindOrCreateFunctionVariable(opt::IRContext* context, opt::Function*,
                                      uint32_t pointer_type_id);

// Returns an OpUndef id from the global value list that is of the given type,
// adding one if it does not exist.
uint32_t FindOrCreateGlobalUndef(opt::IRContext* context, uint32_t type_id);

// Removes any components of |to_block|'s phi instructions relating to
// |from_id|.
void AdaptPhiInstructionsForRemovedEdge(uint32_t from_id,
                                        opt::BasicBlock* to_block);

}  // namespace reduce
}  // namespace spvtools

#endif  // SOURCE_REDUCE_REDUCTION_UTIL_H_
