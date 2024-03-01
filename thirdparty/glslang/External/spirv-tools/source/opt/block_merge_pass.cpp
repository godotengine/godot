// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
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

#include "source/opt/block_merge_pass.h"

#include "source/opt/block_merge_util.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {

bool BlockMergePass::MergeBlocks(Function* func) {
  bool modified = false;
  for (auto bi = func->begin(); bi != func->end();) {
    // Don't bother trying to merge unreachable blocks.
    if (context()->IsReachable(*bi) &&
        blockmergeutil::CanMergeWithSuccessor(context(), &*bi)) {
      blockmergeutil::MergeWithSuccessor(context(), func, bi);
      // Reprocess block.
      modified = true;
    } else {
      ++bi;
    }
  }
  return modified;
}

Pass::Status BlockMergePass::Process() {
  // Process all entry point functions.
  ProcessFunction pfn = [this](Function* fp) { return MergeBlocks(fp); };
  bool modified = context()->ProcessReachableCallTree(pfn);
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

BlockMergePass::BlockMergePass() = default;

}  // namespace opt
}  // namespace spvtools
