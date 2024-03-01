// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
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

#ifndef SOURCE_OPT_BLOCK_MERGE_UTIL_H_
#define SOURCE_OPT_BLOCK_MERGE_UTIL_H_

#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {

// Provides functions for determining when it is safe to merge blocks, and for
// actually merging blocks, for use by various analyses and passes.
namespace blockmergeutil {

// Returns true if and only if |block| has exactly one successor and merging
// this successor into |block| has no impact on the semantics or validity of the
// SPIR-V module.
bool CanMergeWithSuccessor(IRContext* context, BasicBlock* block);

// Requires that |bi| has a successor that can be safely merged into |bi|, and
// performs the merge.
void MergeWithSuccessor(IRContext* context, Function* func,
                        Function::iterator bi);

}  // namespace blockmergeutil
}  // namespace opt
}  // namespace spvtools

#endif  //  SOURCE_OPT_BLOCK_MERGE_UTIL_H_
