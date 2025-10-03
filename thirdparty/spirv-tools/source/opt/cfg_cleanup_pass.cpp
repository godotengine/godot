// Copyright (c) 2017 Google Inc.
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

// This file implements a pass to cleanup the CFG to remove superfluous
// constructs (e.g., unreachable basic blocks, empty control flow structures,
// etc)

#include <queue>
#include <unordered_set>

#include "source/opt/cfg_cleanup_pass.h"

#include "source/opt/function.h"
#include "source/opt/module.h"

namespace spvtools {
namespace opt {

Pass::Status CFGCleanupPass::Process() {
  // Process all entry point functions.
  ProcessFunction pfn = [this](Function* fp) { return CFGCleanup(fp); };
  bool modified = context()->ProcessReachableCallTree(pfn);
  return modified ? Pass::Status::SuccessWithChange
                  : Pass::Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
